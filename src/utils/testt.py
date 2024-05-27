from data_files_managing import save_data_n_labels
from sys import setrecursionlimit
import matplotlib.pyplot as plt
import numpy as np

setrecursionlimit(100000)

class PointCollector:
    def __init__(self, name=None, path='src/utils/Datasets'):
        self.points = {color: [] for color in self.color_map.values()}
        self.current_color = 'blue'
        self.is_drawing = False
        self.eraser_mode = False
        self.fig, self.ax = plt.subplots()
        self.scatter_plots = {color: self.ax.plot([], [], 'o', color=color)[0] for color in self.color_map.values()}  # Line2D objects for each color
        self.create_legend()
        if name is not None:
            self.load_data(f'{path}/{name}_data.txt', f'{path}/{name}_labels.txt')

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
        self.erase_patch = plt.Line2D([0], [0], linestyle="None", marker="", color='black', label="Press 'E' to enable eraser")

        # Place the color legend outside the plot at the top left of the figure, horizontally
        self.legend1 = self.fig.legend(handles=self.legend_patches, loc='upper left', bbox_to_anchor=(0.1, 0.95), markerscale=1, ncol=len(self.legend_patches))

        # Place the save and erase instruction legend outside the plot at the top right of the figure
        self.legend2 = self.fig.legend(handles=[self.save_patch, self.erase_patch], loc='upper right', bbox_to_anchor=(0.9, 0.95), frameon=False)
        self.erase_label = self.legend2.get_texts()[1]  # Reference to the eraser legend text

    def load_data(self, data_path, labels_path):
        data = np.loadtxt(data_path)
        labels = np.loadtxt(labels_path, dtype=int)
        n=len(list(data))
        print(n)
        for point, label in zip(data, labels):
            color = self.color_map[str(label)]
            self.points[color].append(tuple(point))

    def simulate_draw(self,n):
        # Simulate draw action for each loaded point
        i=0
        print(self.points.items())
        for color, points in self.points.items():
            for point in points:
                #print(point)
                i+=1
                #print(i)
                #print(f"Loading dataset {(i/n)*100}%")
                self.current_color = color
                event = type('Event', (object,), {'xdata': point[0], 'ydata': point[1], 'inaxes': True})()
                self.add_point(event)
        self.update_plot()

    def on_press(self, event):
        if event.inaxes:
            self.is_drawing = True
            self.modify_points(event)

    def on_release(self, event):
        self.is_drawing = False

    def on_motion(self, event):
        if self.is_drawing and event.inaxes:
            self.modify_points(event)

    def modify_points(self, event):
        if self.eraser_mode:
            self.erase_point(event)
        else:
            self.add_point(event)
        self.update_plot()

    def add_point(self, event):
        self.points[self.current_color].append((event.xdata, event.ydata))

    def erase_point(self, event):
        erase_radius = 0.005
        for color, points in self.points.items():
            self.points[color] = [(x, y) for x, y in points if (x - event.xdata)**2 + (y - event.ydata)**2 > erase_radius**2]

    def update_plot(self):
        for color, line in self.scatter_plots.items():
            x_data, y_data = zip(*self.points[color]) if self.points[color] else ([], [])
            line.set_data(x_data, y_data)
        self.fig.canvas.draw_idle()

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

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.erase_label.set_text("Press 'E' to disable eraser")
        else:
            self.erase_label.set_text("Press 'E' to enable eraser")
        self.fig.canvas.draw()

def main(name=None):
    collector = PointCollector(name)

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
        elif event.key.lower() == 'e':
            collector.toggle_eraser()

    collector.fig.canvas.mpl_connect('key_press_event', on_key)

    # Adjust the limits to achieve a 3:1 x/y ratio in measurements
    if name is None:
        collector.ax.set_xlim(0, 6)
        collector.ax.set_ylim(0, 2)
    collector.ax.set_xticks([])  # Hide x-axis ticks
    collector.ax.set_yticks([])  # Hide y-axis ticks

    plt.show()

if __name__ == "__main__":
    main()
