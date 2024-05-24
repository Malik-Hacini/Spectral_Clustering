from data_files_managing import save_data_n_labels
from sys import setrecursionlimit
import matplotlib.pyplot as plt

setrecursionlimit(50000)
class PointCollector:
    def __init__(self):
        self.points = {color: [] for color in self.color_map.values()}
        self.current_color = 'blue'
        self.is_drawing = False
        self.fig, self.ax = plt.subplots()
        self.scatter_plots = {color: self.ax.plot([], [], 'o', color=color)[0] for color in self.color_map.values()}  # Line2D objects for each color
        self.create_legend()

    @property
    def color_map(self):
        return {
            '1': 'blue', '2': 'green', '3': 'red', '4': 'cyan',
            '5': 'magenta', '6': 'yellow', '7': 'black',
            '8': 'orange', '9': 'purple'
        }

    def create_legend(self):
        self.legend_patches = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=f'{key}')
            for key, color in self.color_map.items()
        ]
        self.save_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press '0' to save")
        
        # Place the color legend outside the plot at the top left of the figure, horizontally
        self.legend1 = self.fig.legend(handles=self.legend_patches, loc='upper left', bbox_to_anchor=(0.1, 0.95), markerscale=1, ncol=len(self.legend_patches))
        
        # Place the save instruction legend outside the plot at the top right of the figure
        self.legend2 = self.fig.legend(handles=[self.save_patch], loc='upper right', bbox_to_anchor=(0.9, 0.95),frameon=False)

    def on_press(self, event):
        if event.inaxes:
            self.is_drawing = True
            self.add_point(event)

    def on_release(self, event):
        self.is_drawing = False

    def on_motion(self, event):
        if self.is_drawing and event.inaxes:
            self.add_point(event)

    def add_point(self, event):
        self.points[self.current_color].append((event.xdata, event.ydata))
        self.update_plot()

    def update_plot(self):
        for color, line in self.scatter_plots.items():
            x_data, y_data = zip(*self.points[color]) if self.points[color] else ([], [])
            line.set_data(x_data, y_data)
        self.ax.draw_artist(self.ax.patch)
        for line in self.scatter_plots.values():
            self.ax.draw_artist(line)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def generate_data_and_labels(self):
        data = []
        labels = []
        for color, points in self.points.items():
            data.extend(points)
            labels.extend([int(key) for key, val in self.color_map.items() if val == color] * len(points))
        return data, labels

    def change_color(self, key):
        if key in self.color_map:
            self.current_color = self.color_map[key]

def main():
    collector = PointCollector()

    collector.fig.canvas.mpl_connect('button_press_event', collector.on_press)
    collector.fig.canvas.mpl_connect('button_release_event', collector.on_release)
    collector.fig.canvas.mpl_connect('motion_notify_event', collector.on_motion)

    def on_key(event):
        if event.key in collector.color_map:
            collector.change_color(event.key)
        elif event.key == '0':
            data, labels = collector.generate_data_and_labels()
            name = input("Enter the name for the dataset: ")
            save_data_n_labels(data, labels, name)

    collector.fig.canvas.mpl_connect('key_press_event', on_key)

    collector.ax.set_xlim(0, 2)
    collector.ax.set_ylim(0, 2)
    
    plt.show()

if __name__ == "__main__":
    main()
