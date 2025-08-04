import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:convert'; // For utf8 encoding/decoding and json
import 'package:flutter/services.dart' show rootBundle; // For loading tokenizer vocabulary

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hybrid GPT-2 Inference',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const InferenceScreen(),
    );
  }
}

class InferenceScreen extends StatefulWidget {
  const InferenceScreen({Key? key}) : super(key: key);

  @override
  _InferenceScreenState createState() => _InferenceScreenState();
}

class _InferenceScreenState extends State<InferenceScreen> {
  final TextEditingController _promptController = TextEditingController();
  String _status = "Initializing...";
  String _generatedText = "Waiting for generation...";
  bool _isProcessing = false;

  // --- Configure Laptop/Server Connection ---
  // IMPORTANT: Replace with your laptop's actual Wi-Fi IPv4 address!
  static const String LAPTOP_IP = '192.168.1.68'; // e.g., '192.168.1.100'
  static const int SERVER_PORT = 12345;

  // --- GPT-2 Model Specifics ---
  static const int GPT2_VOCAB_SIZE = 50257;
  static const int GPT2_EMBEDDING_DIM = 768;
  static const int MAX_SEQUENCE_LENGTH = 128;

  Interpreter? _interpreter;
  bool _isModelLoaded = false;

  Map<String, int> _vocabMap = {}; // GPT-2 token to ID
  Map<int, String> _idToToken = {}; // ID to GPT-2 token (for debugging/future use)


  @override
  void initState() {
    super.initState();
    _loadTokenizerVocab();
    _loadTFLiteHeadModel();
  }

  // --- Load GPT-2 vocab.json for tokenization ---
  Future<void> _loadTokenizerVocab() async {
    try {
      String jsonString = await rootBundle.loadString('assets/vocab.json');
      Map<String, dynamic> jsonMap = json.decode(jsonString);

      _vocabMap = jsonMap.cast<String, int>();
      _idToToken = _vocabMap.map((k, v) => MapEntry(v, k));
      print("GPT-2 Tokenizer vocabulary loaded successfully.");
    } catch (e) {
      setState(() {
        _status = "Failed to load tokenizer vocab: $e";
      });
      print("Error loading tokenizer vocab: $e");
    }
  }

  // --- Tokenize Prompt using GPT-2 vocabulary ---
  List<int> _tokenizePrompt(String prompt) {
    List<int> tokenIds = [];
    List<String> words = prompt.split(' ');

    int eosTokenId = _vocabMap['<|endoftext|>'] ?? 50256; // Fallback in case vocab isn't fully loaded

    for (String word in words) {
      if (_vocabMap.containsKey(word)) {
        tokenIds.add(_vocabMap[word]!);
      } else {
        for (int i = 0; i < word.length; i++) {
          String char = word[i];
          if (_vocabMap.containsKey(char)) {
            tokenIds.add(_vocabMap[char]!);
          } else {
            tokenIds.add(eosTokenId); // Fallback: Use EOS token for unknown characters
          }
        }
      }
    }

    // Ensure tokens don't exceed MAX_SEQUENCE_LENGTH
    if (tokenIds.length > MAX_SEQUENCE_LENGTH) {
      tokenIds = tokenIds.sublist(0, MAX_SEQUENCE_LENGTH);
    }
    return tokenIds;
  }


  // --- Load TFLite Head Model (GPT-2 Embeddings) ---
  Future<void> _loadTFLiteHeadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model_gpt2_embeddings_head.tflite');
      setState(() {
        _isModelLoaded = true;
        _status = "GPT-2 Embeddings Head Model loaded. Ready.";
      });
      print("TFLite Head Model loaded successfully.");
    } catch (e) {
      setState(() {
        _status = "Failed to load TFLite model: $e";
      });
      print("Error loading TFLite model: $e");
    }
  }

  // --- Run Hybrid Inference ---
  Future<void> _runHybridInference() async {
    if (!_isModelLoaded) {
      setState(() { _status = "Head Model not loaded."; });
      return;
    }
    if (_vocabMap.isEmpty) {
      setState(() { _status = "Tokenizer vocabulary not loaded."; });
      return;
    }

    final prompt = _promptController.text.trim();
    if (prompt.isEmpty) {
      setState(() { _status = "Please enter a prompt."; });
      return;
    }

    setState(() {
      _isProcessing = true;
      _status = "Tokenizing and generating embeddings on phone...";
      _generatedText = "Generating...";
    });

    // 1. Tokenize the prompt (on phone)
    List<int> inputTokenIds = _tokenizePrompt(prompt);

    // 2. Prepare output buffer for intermediate embeddings
    var intermediateEmbeddings = List.generate(
        1,       // Batch size
          (i) => List.generate(
        inputTokenIds.length, // Sequence length
            (j) => List.filled(
          GPT2_EMBEDDING_DIM,
          0.0,
        ),
      ),
    );

    // 3. Run Head Model Inference (on phone)
    try {
      _interpreter!.run([inputTokenIds], intermediateEmbeddings);
      print("Head model (embeddings) inference complete. Intermediate embeddings generated.");

    } catch (e) {
      setState(() { _status = "Error running head model: $e"; });
      print("TFLite run error: $e");
      _isProcessing = false;
      return;
    }

    // 4. Prepare data for transmission: JSON with embeddings and attention mask
    List<int> attentionMask = List.filled(inputTokenIds.length, 1);

    Map<String, dynamic> dataPayload = {
      'input_embeddings': intermediateEmbeddings,
      'attention_mask': attentionMask,
    };
    String jsonString = json.encode(dataPayload);
    Uint8List dataToSend = utf8.encode(jsonString);

    setState(() {
      _status = "Sending embeddings and mask to laptop (Full GPT-2 Model)...";
    });

    // --- Network Communication and Server Inference ---
    await _sendDataToServer(dataToSend, prompt);

    setState(() {
      _isProcessing = false;
    });
  }

  // --- TCP Communication Logic (Sends/Receives JSON) ---
  Future<void> _sendDataToServer(Uint8List dataToSend, String prompt) async {
    Socket? socket;
    try {
      socket = await Socket.connect(LAPTOP_IP, SERVER_PORT, timeout: const Duration(seconds: 15));
      print('Connected to server: ${socket.remoteAddress.address}:${socket.remotePort}');

      // Send the size of the JSON payload first (4 bytes)
      final ByteData dataSize = ByteData(4)..setUint32(0, dataToSend.length, Endian.big);
      socket.add(dataSize.buffer.asUint8List());

      // Send the JSON payload
      socket.add(dataToSend);
      await socket.flush();

      setState(() { _status = "Awaiting generated text from laptop..."; });

      // This buffer will accumulate all incoming bytes
      List<int> allReceivedBytes = [];

      // We need to read exactly 4 bytes for the response size first
      Uint8List responseSizeBuffer = Uint8List(4);
      int bytesReadForSize = 0;

      // Use a stream listener to collect bytes for the size prefix
      await for (var chunk in socket) {
        // Add current chunk to our overall buffer
        allReceivedBytes.addAll(chunk);

        // Check if we have enough bytes for the size
        if (allReceivedBytes.length >= 4) {
          responseSizeBuffer.setRange(0, 4, allReceivedBytes.sublist(0, 4));
          bytesReadForSize = 4;
          break; // Got the size, exit the loop
        }
      }

      if (bytesReadForSize < 4) {
        throw Exception("Failed to read full response size from server.");
      }

      int responseSize = ByteData.view(responseSizeBuffer.buffer).getUint32(0, Endian.big);

      // Now, ensure we have enough bytes for the actual response data
      // We already have some bytes in allReceivedBytes from the size reading phase
      while (allReceivedBytes.length < (4 + responseSize)) { // 4 for size, plus responseSize for data
        List<int> chunk = await socket.where((data) => data.isNotEmpty).first;
        allReceivedBytes.addAll(chunk);
      }

      // Extract the actual generated text bytes (after the 4-byte size prefix)
      Uint8List generatedTextBytes = Uint8List.fromList(allReceivedBytes.sublist(4, 4 + responseSize));

      final String generatedText = utf8.decode(generatedTextBytes);
      // --- End of corrected data receiving logic ---

      setState(() {
        _status = "Generation complete.";
        _generatedText = prompt + generatedText;
      });

    } on SocketException catch (e) {
      setState(() {
        _status = "Connection failed: $e. Check IP/Port and server status.";
      });
      print("Socket connection error: $e");
    } catch (e) {
      setState(() {
        _status = "An error occurred: $e";
      });
      print("General error: $e");
    } finally {
      socket?.close();
    }
  }

  @override
  void dispose() {
    _promptController.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hybrid GPT-2 Inference PoC'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text(
              _status,
              style: TextStyle(color: _isProcessing ? Colors.blue : Colors.green[800], fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _promptController,
              decoration: const InputDecoration(
                labelText: 'Enter your prompt',
                border: OutlineInputBorder(),
                hintText: 'E.g., Once upon a time,',
              ),
              maxLines: 4,
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isProcessing || !_isModelLoaded || _vocabMap.isEmpty ? null : _runHybridInference,
              child: _isProcessing
                  ? const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(width: 10),
                  Text("Processing..."),
                ],
              )
                  : const Text('Generate Text (Hybrid GPT-2)'),
            ),
            const SizedBox(height: 30),
            const Text(
              'Generated Text:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                  color: Colors.grey[100],
                ),
                child: SingleChildScrollView(
                  child: Text(
                    _generatedText,
                    style: const TextStyle(fontSize: 16),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}