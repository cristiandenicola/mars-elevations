import numpy as np
import matplotlib.pyplot as plt

# Carica i dati di loss dal file .npy
loss_values = np.load('/Users/cristiandenicola/Projects/mars-elevations/loss_curve.npy')

# Crea una lista di epoche (1, 2, 3, ...)
epochs = np.arange(1, len(loss_values) + 1)

# Calcola la percentuale di miglioramento
improvement = ((loss_values[0] - loss_values[-1]) / loss_values[0]) * 100

# Crea la figura e gli assi
plt.figure(figsize=(10, 6))

# Stile più moderno
plt.style.use('seaborn-v0_8-darkgrid')

# Plotta i dati con un colore e stile eleganti
plt.plot(epochs, loss_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7, color='#4C72B0')

# Aggiungi un'area ombreggiata sotto la curva per enfatizzare il trend
plt.fill_between(epochs, loss_values, min(loss_values)*0.95, alpha=0.1, color='#4C72B0')

# Etichette e titolo con più dettagli
plt.xlabel('Epoche', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Andamento Loss durante il Training', fontsize=14, fontweight='bold')

# Aggiungi annotazione con la percentuale di miglioramento
plt.annotate(f'Miglioramento totale: {improvement:.2f}%\n'
             f'Da {loss_values[0]:.2f} a {loss_values[-1]:.2f}',
             xy=(epochs[-1], loss_values[-1]),
             xytext=(epochs[-1]*0.8, loss_values[0]*0.6),
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
             fontsize=10)

# Aggiungi griglia per migliorare la leggibilità
plt.grid(True, linestyle='--', alpha=0.7)

# Aggiungi riquadro attorno al grafico
plt.box(True)

# Miglioramenti estetici aggiuntivi
plt.tight_layout()

# Mostra il grafico
plt.show()

plt.savefig('training_loss_01_05_25.png', dpi=300, bbox_inches='tight')