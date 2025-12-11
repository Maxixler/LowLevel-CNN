import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEEP LEARNING MOTORU
# =============================================================================

class Dense:
    def __init__(self, input_size, output_size, learning_rate=0.01, momentum=0.9):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((output_size, 1))
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.momentum = momentum
        self.lr = learning_rate

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.W, self.input) + self.b

    def backward(self, output_gradient):
        self.dW = np.dot(output_gradient, self.input.T)
        self.db = output_gradient
        return np.dot(self.W.T, output_gradient)

    def update(self):
        self.v_W = (self.momentum * self.v_W) - (self.lr * self.dW)
        self.v_b = (self.momentum * self.v_b) - (self.lr * self.db)
        self.W += self.v_W
        self.b += self.v_b

class ReLU:
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)

class Softmax:
    def forward(self, input_data):
        exps = np.exp(input_data - np.max(input_data))
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output
    def backward(self, output_gradient):
        return output_gradient

def one_hot_encode(y, num_classes=4):
    one_hot = np.zeros((num_classes, 1))
    one_hot[y] = 1
    return one_hot

# =============================================================================
# 2. TAMAMEN YENİ VERİ SETİ - GERÇEK EN KISA YOL
# =============================================================================

def generate_smart_navigation_data(num_samples, grid_size=10):
    """
    Gerçek en kısa yolu öğreten AKILLI veri seti
    """
    X = []
    Y = []
    
    for _ in range(num_samples):
        robot_x, robot_y = np.random.randint(0, grid_size, 2)
        target_x, target_y = np.random.randint(0, grid_size, 2)
        
        # Aynı noktadaysa atla
        if robot_x == target_x and robot_y == target_y:
            continue
            
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        input_vec = np.array([[dx / grid_size], [dy / grid_size]])
        
        # EN ÖNEMLİ DEĞİŞİKLİK: Gerçek optimal yönü hesapla
        # Öncelikle hangi eksende daha fazla ilerleme needed
        optimal_action = -1
        
        # Eğer x yönünde daha fazla mesafe varsa
        if abs(dx) > abs(dy):
            optimal_action = 0 if dx > 0 else 1  # SAĞ veya SOL
        # Eğer y yönünde daha fazla mesafe varsa
        elif abs(dy) > abs(dx):
            optimal_action = 2 if dy > 0 else 3  # YUKARI veya AŞAĞI
        else:
            # Eşit mesafe varsa rastgele seç (doğal yolu taklit et)
            if dx != 0 and dy != 0:
                optimal_action = np.random.choice([0 if dx > 0 else 1, 2 if dy > 0 else 3])
            elif dx == 0:
                optimal_action = 2 if dy > 0 else 3
            else:
                optimal_action = 0 if dx > 0 else 1
        
        X.append(input_vec)
        Y.append(optimal_action)
        
    return X, Y

# =============================================================================
# 3. GELİŞMİŞ EĞİTİM STRATEJİSİ
# =============================================================================

# Daha büyük ve daha akıllı veri seti
print("AKILLI Robot Eğitiliyor...")
X_train, Y_labels = generate_smart_navigation_data(15000, grid_size=20)
Y_train = [one_hot_encode(y) for y in Y_labels]

# Etiket dağılımını kontrol et
unique, counts = np.unique(Y_labels, return_counts=True)
print("Etiket dağılımı:", dict(zip(['SAĞ', 'SOL', 'YUKARI', 'AŞAĞI'], counts)))

loss_history = []
accuracy_history = []

# Daha iyi mimari
network = [
    Dense(2, 32, learning_rate=0.005),
    ReLU(),
    Dense(32, 16, learning_rate=0.005),
    ReLU(), 
    Dense(16, 8, learning_rate=0.005),
    ReLU(),
    Dense(8, 4, learning_rate=0.005),
    Softmax()
]

# Eğitim döngüsü
for epoch in range(300):
    total_loss = 0
    correct_predictions = 0
    
    for x, y_true in zip(X_train, Y_train):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)
            
        # Loss
        loss = -np.sum(y_true * np.log(output + 1e-9))
        total_loss += loss
        
        # Accuracy
        if np.argmax(output) == np.argmax(y_true):
            correct_predictions += 1
        
        # Backward
        grad = output - y_true
        for layer in reversed(network):
            grad = layer.backward(grad)
            
        # Update
        for layer in network:
            if hasattr(layer, 'update'): 
                layer.update()
                
    avg_loss = total_loss / len(X_train)
    accuracy = correct_predictions / len(X_train)
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)
    
    if (epoch+1) % 30 == 0:
        print(f"Epoch {epoch+1}/300 - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.3f}")

# Grafikler
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(loss_history)
ax1.set_title("Eğitim Kaybı")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

ax2.plot(accuracy_history)
ax2.set_title("Eğitim Doğruluğu")
ax2.set_xlabel("Epoch") 
ax2.set_ylabel("Accuracy")
plt.show()

# =============================================================================
# 4. AKILLI SİMÜLASYON
# =============================================================================

def smart_simulate_robot(start_pos, target_pos, grid_size=20):
    path_x = [start_pos[0]]
    path_y = [start_pos[1]]
    
    current_pos = list(start_pos)
    steps = 0
    max_steps = grid_size * 2
    visited_positions = set([tuple(start_pos)])
    
    print(f"\n🚀 AKILLI Rota: {start_pos} -> {target_pos}")
    
    while (current_pos[0] != target_pos[0] or current_pos[1] != target_pos[1]) and steps < max_steps:
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Eğer zaten hedefteysek çık
        if dx == 0 and dy == 0:
            break
            
        inp = np.array([[dx / grid_size], [dy / grid_size]])
        
        # Model tahmini
        out = inp
        for layer in network:
            out = layer.forward(out)
        
        action = np.argmax(out)
        action_probs = out.flatten()
        
        # Hareketi uygula
        move_name = ""
        old_pos = current_pos.copy()
        
        if action == 0 and current_pos[0] < grid_size - 1:  # SAĞ
            current_pos[0] += 1
            move_name = "➡️ SAĞ"
        elif action == 1 and current_pos[0] > 0:  # SOL  
            current_pos[0] -= 1
            move_name = "⬅️ SOL"
        elif action == 2 and current_pos[1] < grid_size - 1:  # YUKARI
            current_pos[1] += 1
            move_name = "⬆️ YUKARI"
        elif action == 3 and current_pos[1] > 0:  # AŞAĞI
            current_pos[1] -= 1
            move_name = "⬇️ AŞAĞI"
        else:
            move_name = "🚫 SINIR"
        
        # Eğer hareket ettiysek ve bu pozisyonu daha önce görmediysek
        if current_pos != old_pos:
            if tuple(current_pos) in visited_positions:
                # Döngüden kaçın - rastgele farklı bir hareket dene
                current_pos = old_pos.copy()
                continue
                
            visited_positions.add(tuple(current_pos))
            path_x.append(current_pos[0])
            path_y.append(current_pos[1])
            
        steps += 1
        print(f"Adım {steps}: {move_name} -> {current_pos} (Hedef: {target_pos})")
        
        if steps >= max_steps:
            print("⏰ Maksimum adım aşıldı!")
            break

    # Performans analizi
    success = current_pos[0] == target_pos[0] and current_pos[1] == target_pos[1]
    optimal_steps = abs(start_pos[0]-target_pos[0]) + abs(start_pos[1]-target_pos[1])
    efficiency = optimal_steps / steps if steps > 0 else 0
    
    print(f"\n{'✅ BAŞARILI' if success else '❌ BAŞARISIZ'}")
    print(f"Adım sayısı: {steps} (Optimal: {optimal_steps})")
    print(f"Verimlilik: {efficiency:.2%}")

    # Görselleştirme
    plt.figure(figsize=(10, 10))
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, grid_size+1)
    plt.ylim(-1, grid_size+1)
    
    # Izgara çizgileri
    for i in range(grid_size + 1):
        plt.axhline(i, color='gray', linestyle='-', alpha=0.2)
        plt.axvline(i, color='gray', linestyle='-', alpha=0.2)
    
    plt.scatter(start_pos[0], start_pos[1], c='blue', s=300, label='Başlangıç 🤖', edgecolors='black')
    plt.scatter(target_pos[0], target_pos[1], c='red', s=300, marker='*', label='Hedef 🎯', edgecolors='black')
    
    plt.plot(path_x, path_y, c='green', linewidth=3, linestyle='-', label='AI Rotası', marker='o', markersize=6)
    
    plt.legend(fontsize=12)
    plt.title(f"Robot Yolu - {steps} adım ({'✅ BAŞARILI' if success else '❌ BAŞARISIZ'})", fontsize=14)
    plt.show()
    
    return success, steps, efficiency

# ÇOKLU TEST
print("=" * 50)
print("🤖 ROBOT NAVİGASYON TESTLERİ")
print("=" * 50)

test_cases = [
    ((2, 2), (15, 18), "Çapraz yol"),
    ((0, 0), (19, 19), "Köşeden köşeye"), 
    ((5, 15), (15, 5), "Çapraz ters"),
    ((10, 10), (18, 12), "Kısa yol"),
    ((3, 17), (17, 3), "Uzun çapraz")
]

results = []
for start, target, desc in test_cases:
    print(f"\n🧪 TEST: {desc}")
    success, steps, efficiency = smart_simulate_robot(start, target)
    results.append((desc, success, steps, efficiency))

# Sonuç özeti
print("\n" + "=" * 50)
print("📊 TEST SONUÇLARI")
print("=" * 50)
for desc, success, steps, efficiency in results:
    status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
    print(f"{desc}: {status} - {steps} adım - Verimlilik: {efficiency:.2%}")