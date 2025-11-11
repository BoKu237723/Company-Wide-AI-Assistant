import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import json
import datetime
from collections import defaultdict
import os

class ExplicitRNNCell(nn.Module):
    """
    Explicit RNN Cell showing all components:
    h = hidden state
    W = weight matrices
    b = bias
    x = input
    y = output
    Whx = weight for input xt
    Whh = weight for previous hidden state ht-1
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.Whx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wyh = nn.Linear(hidden_dim, output_dim)
        
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        self.b_y = nn.Parameter(torch.zeros(output_dim))
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x, h_prev=None):
        """
        Forward pass with explicit RNN equations:
        h_t = tanh(Whx * x_t + Whh * h_{t-1} + b_h)
        y_t = Wyh * h_t + b_y
        """
        batch_size = x.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim)

        # h = tanh(Whx * x + Whh * h_prev + b_h)
        h_current = torch.tanh(
            self.Whx(x) +      # Whx * x_t
            self.Whh(h_prev) + # Whh * h_{t-1}  
            self.b_h           # b_h
        )
        
        # y = Wyh * h + b_y
        y_output = self.Wyh(h_current) + self.b_y
        
        return y_output, h_current

class DepartmentClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn_cells = nn.ModuleList([
            ExplicitRNNCell(embedding_dim, hidden_dim, hidden_dim) 
            for _ in range(n_layers)
        ])
        
        self.Wyh_final = nn.Linear(hidden_dim, output_dim)
        self.b_y_final = nn.Parameter(torch.zeros(output_dim))
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
    def forward(self, x, sequence_lengths=None):
        batch_size, seq_len = x.size()
        
        x_embedded = self.embedding(x)
        
        hidden_states = [torch.zeros(batch_size, self.hidden_dim) for _ in range(self.n_layers)]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x_embedded[:, t, :]
            
            for layer_idx, rnn_cell in enumerate(self.rnn_cells):
                h_prev = hidden_states[layer_idx]
                
                y_layer, h_current = rnn_cell(layer_input, h_prev)
                
                hidden_states[layer_idx] = h_current
                layer_input = h_current  
            
            outputs.append(y_layer)
        
        outputs = torch.stack(outputs).transpose(0, 1)
        
        if sequence_lengths is not None:
            last_outputs = []
            for i, length in enumerate(sequence_lengths):
                last_outputs.append(outputs[i, length-1, :])
            final_hidden = torch.stack(last_outputs)
        else:
            final_hidden = outputs[:, -1, :]
        
        final_output = self.Wyh_final(final_hidden) + self.b_y_final
        
        return final_output

class QuestionDataset(Dataset):
    def __init__(self, questions, departments, tokenizer, max_length=100):
        self.questions = questions
        self.departments = departments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dept_to_idx = {dept: i for i, dept in enumerate(['finance', 'marketing', 'IT'])}
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        department = self.departments[idx]
        
        tokens = self.tokenizer(question)[:self.max_length]
        tokens += [0] * (self.max_length - len(tokens)) 
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.dept_to_idx[department], dtype=torch.long),
            'length': torch.tensor(min(len(self.tokenizer(question)), self.max_length), dtype=torch.long)
        }

class DepartmentClassifier:
    def __init__(self):
        self.departments = ['finance', 'marketing', 'IT']
        self.rnn_model = None
        self.vocab = None
        self.model_dir = 'rnn_models'
        
    def ensure_model_dir(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"📁 Created model directory: {self.model_dir}")
    
    def get_model_paths(self):
        self.ensure_model_dir()
        return {
            'model': os.path.join(self.model_dir, 'department_classifier_rnn.pth'),
            'vocab': os.path.join(self.model_dir, 'vocab.pth'),
            'training_data': os.path.join(self.model_dir, 'training_data.json'),
            'model_info': os.path.join(self.model_dir, 'model_info.json'),
            'performance': os.path.join(self.model_dir, 'performance_history.json')
        }
    
    def setup_rnn_model(self):
        paths = self.get_model_paths()
    
        try:
            if all(os.path.exists(path) for path in [paths['model'], paths['vocab'], paths['model_info']]):
                self.rnn_model = torch.load(paths['model'], map_location='cpu', weights_only=False)
                self.vocab = torch.load(paths['vocab'])
            
                with open(paths['model_info'], 'r') as f:
                    model_info = json.load(f)
            
                print("✅ Loaded pre-trained RNN classifier")
                print(f"   📅 Trained on: {model_info.get('training_date', 'Unknown')}")
                print(f"   📊 Training samples: {model_info.get('training_samples', 0)}")
                print(f"   🎯 Accuracy: {model_info.get('accuracy', 'Unknown')}")
            
                self.print_model_parameters()
            
            else:
                print("🔄 No complete pre-trained RNN model found")
                print("   Initializing new model...")
                self.initialize_new_model()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Initializing new model...")
            self.initialize_new_model()

    def initialize_new_model(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.rnn_model = None
    
    def print_model_parameters(self):
        if self.rnn_model:
            print("\n📊 RNN Model Parameters:")
            total_params = 0
            for name, param in self.rnn_model.named_parameters():
                if param.requires_grad:
                    print(f"   {name}: {tuple(param.shape)}")
                    total_params += param.numel()
            print(f"   Total trainable parameters: {total_params:,}")
    
    def build_vocab(self, texts):
        word_freq = defaultdict(int)
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                word_freq[word] += 1

        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, freq) in enumerate(
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10000]
        ):
            self.vocab[word] = idx + 2
    
    def text_to_sequence(self, text):
        if not self.vocab:
            return []
        
        words = re.findall(r'\w+', text.lower())
        return [self.vocab.get(word, 1) for word in words if word in self.vocab]  # 1 for UNK
    
    def robust_json_extraction(self, text):
        json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|(\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1) or json_match.group(0)
                return json.loads(json_str)
            except:
                pass
        
        departments_match = re.search(r'\"departments\"\s*:\s*\[.*\]', text, re.DOTALL)
        if departments_match:
            try:
                json_str = '{' + departments_match.group(0) + '}'
                return json.loads(json_str)
            except:
                pass
        
        departments = []
        dept_pattern = r'[\"\']?department[\"\']?\s*:\s*[\"\'](\w+)[\"\']'
        conf_pattern = r'[\"\']?confidence[\"\']?\s*:\s*(\d+\.\d+|\d+)'
        reason_pattern = r'[\"\']?reason[\"\']?\s*:\s*[\"\"]([^\"]+?)[\"\"]'
        
        dept_matches = re.findall(dept_pattern, text, re.IGNORECASE)
        conf_matches = re.findall(conf_pattern, text)
        reason_matches = re.findall(reason_pattern, text)
        
        for i, dept in enumerate(dept_matches):
            if i < len(conf_matches):
                try:
                    confidence = float(conf_matches[i])
                    reason = reason_matches[i] if i < len(reason_matches) else "AI determined relevance"
                    
                    if dept.lower() in [d.lower() for d in self.departments]:
                        actual_dept = next(d for d in self.departments if d.lower() == dept.lower())
                        departments.append({
                            'department': actual_dept,
                            'confidence': min(1.0, max(0.0, confidence)),
                            'reason': reason
                        })
                except ValueError:
                    continue
        
        if departments:
            return {'departments': departments}
        
        return None

    def rnn_classify_department(self, question):
        if not self.rnn_model:
            return None
        
        self.rnn_model.eval()
        with torch.no_grad():
            sequence = self.text_to_sequence(question)
            if not sequence:
                return None
                
            max_length = 50
            if len(sequence) < max_length:
                sequence = sequence + [0] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
                
            tokens = torch.tensor([sequence], dtype=torch.long)
            length = torch.tensor([min(len(self.text_to_sequence(question)), max_length)], dtype=torch.long)
            
            output = self.rnn_model(tokens, length)
            probabilities = torch.softmax(output, dim=1)
            
            confidences = probabilities.squeeze().tolist()
            results = []
            
            for i, dept in enumerate(self.departments):
                results.append({
                    'department': dept,
                    'confidence': confidences[i],
                    'reason': f'RNN classification: Whx*x + Whh*h + b_h → tanh → Wyh*h + b_y = {confidences[i]:.3f}',
                    'method': 'RNN'
                })
            
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return {'departments': results, 'method': 'RNN'}

    def classify_department(self, question):
        import ollama
        
        classification_prompt = f"""
        Analyze this question and determine which department(s) it relates to. Use contextual understanding and semantic analysis.
        
        Available departments: {', '.join(self.departments)}
        
        IMPORTANT: Return ONLY valid JSON in this exact format:
        {{
            "departments": [
                {{
                    "department": "finance", 
                    "confidence": 0.85, 
                    "reason": "The question mentions budget and financial planning which falls under finance"
                }}
            ]
        }}
        
        CRITICAL RULES:
        1. Confidence scores MUST be decimal numbers between 0.0 and 1.0
        2. Include ALL relevant departments, even with low confidence
        3. Sort by confidence score (highest first)
        4. Use semantic understanding - don't rely on exact keyword matching
        5. Handle typos and informal language intelligently
        6. Consider context and intent, not just surface-level words
        7. If unsure, provide lower confidence scores rather than excluding
        
        Question to analyze: "{question}"
        
        Think step by step about:
        - What is the user really asking about?
        - Which department's responsibilities match this need?
        - Are there multiple angles to this question?
        - What would be the consequences of missing a relevant department?
        """
        
        try:
            print("🎯 Analyzing question with AI (semantic understanding)...")
            response = ollama.chat(model='llama3.1:8b', messages=[
                {
                    'role': 'system',
                    'content': 'You are a department classification expert. Always return valid JSON. Understand context and intent, not just keywords.'
                },
                {
                    'role': 'user',
                    'content': classification_prompt
                }
            ])
            
            content = response['message']['content']
            print(f"🤖 Raw AI classification response:\n{content}\n")
            
            try:
                classification_data = json.loads(content)
                if 'departments' in classification_data:
                    classification_data['departments'].sort(key=lambda x: x['confidence'], reverse=True)
                    return classification_data
            except json.JSONDecodeError:
                pass
            
            classification_data = self.robust_json_extraction(content)
            if classification_data and 'departments' in classification_data:
                classification_data['departments'].sort(key=lambda x: x['confidence'], reverse=True)
                return classification_data
            
            print("⚠️ Using final fallback classification")
            departments = []
            for dept in self.departments:
                departments.append({
                    'department': dept,
                    'confidence': 0.2,
                    'reason': "AI classification failed - using equal distribution"
                })
            return {'departments': departments}
                
        except Exception as e:
            print(f"❌ Error in ML classification: {e}")
            departments = []
            for dept in self.departments:
                departments.append({
                    'department': dept,
                    'confidence': 0.1,
                    'reason': "Classification error occurred"
                })
            return {'departments': departments}

    def enhanced_classify_department(self, question):
        rnn_result = self.rnn_classify_department(question)
        
        if rnn_result and rnn_result['departments'][0]['confidence'] > 0.6:
            print("🎯 Using RNN classification (mathematically derived)")
            print(f"   RNN Equation: h_t = tanh(Whx*x_t + Whh*h_{{t-1}} + b_h)")
            print(f"   Output Equation: y = Wyh*h + b_y")
            return rnn_result
        else:
            print("🔄 RNN uncertain, using LLM with RNN insights")
            llm_result = self.classify_department(question)
            
            if rnn_result:
                self.combine_rnn_llm_insights(llm_result, rnn_result)
            
            return llm_result

    def combine_rnn_llm_insights(self, llm_result, rnn_analysis):
        rnn_confidences = {dept['department']: dept['confidence'] 
                          for dept in rnn_analysis['departments']}
        
        for dept_info in llm_result['departments']:
            dept = dept_info['department']
            if dept in rnn_confidences:
                rnn_conf = rnn_confidences[dept]
                blended_conf = 0.3 * rnn_conf + 0.7 * dept_info['confidence']
                dept_info['confidence'] = blended_conf
                dept_info['reason'] += f" | RNN confidence: {rnn_conf:.3f}"

    def save_model_complete(self, training_data=None, accuracy=None):
        if not self.rnn_model or not self.vocab:
            print("❌ No model to save")
            return False
        
        paths = self.get_model_paths()
        
        try:
            torch.save(self.rnn_model, paths['model'])
            torch.save(self.vocab, paths['vocab'])
            
            model_info = {
                'training_date': datetime.datetime.now().isoformat(),
                'training_samples': len(training_data) if training_data else 0,
                'vocabulary_size': len(self.vocab),
                'accuracy': accuracy,
                'departments': self.departments,
                'model_architecture': 'DepartmentClassifierRNN'
            }
            
            with open(paths['model_info'], 'w') as f:
                json.dump(model_info, f, indent=2)
            
            if training_data:
                with open(paths['training_data'], 'w') as f:
                    json.dump(training_data, f, indent=2)
            
            print("💾 Model saved successfully!")
            print(f"   📁 Location: {self.model_dir}")
            print(f"   📊 Vocabulary size: {len(self.vocab)}")
            print(f"   📅 Saved on: {model_info['training_date']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def train_rnn_with_saving(self, training_data, validation_data=None):
        self.build_vocab([item['question'] for item in training_data])
        
        if self.rnn_model is None:
            vocab_size = len(self.vocab)
            self.rnn_model = DepartmentClassifierRNN(
                vocab_size=vocab_size,
                embedding_dim=100,
                hidden_dim=128,
                output_dim=len(self.departments)
            )
        
        dataset = QuestionDataset(
            [item['question'] for item in training_data],
            [item['department'] for item in training_data],
            self.text_to_sequence
        )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)
        
        print("🧠 Training RNN with Auto-Save...")
        print("   RNN Equation: h_t = tanh(Whx*x_t + Whh*h_{t-1} + b_h)")
        print("   Output Equation: y = Wyh*h + b_y")
        
        best_accuracy = 0
        training_history = []
        
        for epoch in range(10):
            self.rnn_model.train()
            total_loss = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                predictions = self.rnn_model(batch['tokens'], batch['length'])
                loss = criterion(predictions, batch['label'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            accuracy = self.evaluate_model(validation_data) if validation_data else 0
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            print(f'   Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model_complete(training_data, accuracy)
                print(f"   💾 New best model saved! Accuracy: {accuracy:.3f}")
        
        self.save_training_history(training_history)
        
        return best_accuracy

    def evaluate_model(self, validation_data):
        if not validation_data:
            return 0
        
        correct = 0
        total = 0
        
        self.rnn_model.eval()
        with torch.no_grad():
            for item in validation_data:
                question = item['question']
                true_department = item['department']
                
                rnn_result = self.rnn_classify_department(question)
                if rnn_result and rnn_result['departments']:
                    predicted_department = rnn_result['departments'][0]['department']
                    if predicted_department == true_department:
                        correct += 1
                total += 1
        
        return correct / total if total > 0 else 0

    def save_training_history(self, history):
        paths = self.get_model_paths()
        
        existing_history = []
        if os.path.exists(paths['performance']):
            try:
                with open(paths['performance'], 'r') as f:
                    existing_history = json.load(f)
            except:
                existing_history = []
        
        existing_history.extend(history)
        
        with open(paths['performance'], 'w') as f:
            json.dump(existing_history, f, indent=2)
        
        print(f"📈 Training history saved ({len(history)} new records)")

    def continuous_learning(self, new_interactions):
        if not new_interactions:
            return
        
        paths = self.get_model_paths()
        
        existing_data = []
        if os.path.exists(paths['training_data']):
            try:
                with open(paths['training_data'], 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        training_data = []
        for interaction in new_interactions:
            question = interaction['question']
            if 'answers' in interaction and interaction['answers']:
                best_department = max(interaction['answers'].items(), 
                                    key=lambda x: x[1].get('confidence', 0))[0]
                
                training_data.append({
                    'question': question,
                    'department': best_department,
                    'timestamp': interaction.get('timestamp', datetime.datetime.now().isoformat())
                })
        
        all_questions = {item['question'] for item in existing_data}
        new_unique_data = [item for item in training_data if item['question'] not in all_questions]
        
        if not new_unique_data:
            print("🤷 No new unique training data")
            return
        
        updated_data = existing_data + new_unique_data
        
        print(f"🔄 Continuous learning with {len(new_unique_data)} new samples")
        print(f"   Total training samples: {len(updated_data)}")
        
        if len(updated_data) >= 10:
            split_idx = int(0.8 * len(updated_data))
            train_data = updated_data[:split_idx]
            val_data = updated_data[split_idx:]
            
            accuracy = self.train_rnn_with_saving(train_data, val_data)
            print(f"✅ Continuous learning complete! Accuracy: {accuracy:.3f}")
            
            with open(paths['training_data'], 'w') as f:
                json.dump(updated_data, f, indent=2)
        else:
            print("📊 Collecting more data before retraining...")
            with open(paths['training_data'], 'w') as f:
                json.dump(updated_data, f, indent=2)

    def load_or_create_initial_data(self):
        paths = self.get_model_paths()
        
        if os.path.exists(paths['training_data']):
            try:
                with open(paths['training_data'], 'r') as f:
                    return json.load(f)
            except:
                pass
        
        initial_data = [
            {'question': 'What was our marketing budget?', 'department': 'marketing'},
            {'question': 'How much did we spend on ads?', 'department': 'marketing'},
            {'question': 'Show me financial reports', 'department': 'finance'},
            {'question': 'Revenue last quarter', 'department': 'finance'},
            {'question': 'Server uptime statistics', 'department': 'IT'},
            {'question': 'Network security updates', 'department': 'IT'},
            {'question': 'Social media campaign results', 'department': 'marketing'},
            {'question': 'IT infrastructure costs', 'department': 'finance'},
            {'question': 'Marketing campaign performance', 'department': 'marketing'},
            {'question': 'Budget allocation for IT', 'department': 'finance'},
            {'question': 'Website traffic analysis', 'department': 'marketing'},
            {'question': 'Financial statements review', 'department': 'finance'},
            {'question': 'System maintenance schedule', 'department': 'IT'},
            {'question': 'Advertising spend breakdown', 'department': 'marketing'},
            {'question': 'Profit and loss statement', 'department': 'finance'},
        ]
        
        print("📝 Created initial training dataset")
        return initial_data

    def print_model_status(self):

        paths = self.get_model_paths()
        
        print("\n📊 Model Status:")
        print(f"   RNN Model: {'✅ Loaded' if self.rnn_model else '❌ Not trained'}")
        
        if os.path.exists(paths['training_data']):
            with open(paths['training_data'], 'r') as f:
                data = json.load(f)
            print(f"   Training samples: {len(data)}")
        
        if os.path.exists(paths['model_info']):
            with open(paths['model_info'], 'r') as f:
                info = json.load(f)
            print(f"   Vocabulary size: {info.get('vocabulary_size', 'Unknown')}")
            print(f"   Last trained: {info.get('training_date', 'Unknown')}")
            print(f"   Best accuracy: {info.get('accuracy', 'Unknown')}")