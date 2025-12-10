import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def visualize_training(data_file, history_file):
    try:
        dataset = np.loadtxt(data_file)
        X = dataset[:, 0:2]
        y = dataset[:, 2]
    except Exception as e:
        print(f"Error reading data: {e}")
        return
    history = []
    try:
        with open(history_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                history.append([
                    int(parts[0]),     
                    int(parts[1]),     
                    float(parts[2]),    
                    float(parts[3]),   
                    float(parts[4]),
                    int(parts[5])
                ])
    except Exception as e:
        print(f"Error reading history: {e}")
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title('Perceptron: Linear Binary Classification')    
    class_1 = X[y == 1]
    class_minus_1 = X[y == -1]
    ax.scatter(class_1[:, 0], class_1[:, 1], color='blue', marker='o', s=100, alpha=0.6, label='Class 1')
    ax.scatter(class_minus_1[:, 0], class_minus_1[:, 1], color='red', marker='s', s=100, alpha=0.6, label='Class -1')   
    current_point_highlight, = ax.plot([], [], 'o', color='yellow', markersize=15, markeredgecolor='black', markeredgewidth=2, alpha=0.9, label='Current Point')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    line, = ax.plot([], [], 'g-', linewidth=3, label='Decision Boundary')
    text_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=11, fontfamily='monospace',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f"Perceptron Training on {len(X)} points")
    def update(frame_idx):
        epoch, point_idx, w1, w2, b, total_updates = history[frame_idx]
        x_vals = np.array([x_min, x_max])
        if w2 != 0:
            y_vals = (-w1 * x_vals - b) / w2
            line.set_data(x_vals, y_vals)
        else:
            line.set_data([-b/w1, -b/w1], [y_min, y_max])
        if point_idx < len(X):
            current_point_highlight.set_data([X[point_idx, 0]], [X[point_idx, 1]])
        status_text = (
            f"Epoch       : {epoch}\n"
            f"Point Index : {point_idx}\n"
            f"Weights     : [{w1:.2f}, {w2:.2f}]\n"
            f"Bias        : {b:.2f}\n"
            f"----------------------------\n"
            f"Total Updates: {total_updates}"
        )
        text_info.set_text(status_text)  
        return line, text_info, current_point_highlight
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=30, blit=True, repeat=False)

    plt.show()
