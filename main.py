from PIL import Image
import numpy as np
import os
import math
from heapq import heappush, heappop
import logging
import sys
import getopt


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class SeamCarving:

    def __init__(self, path):
        self.image_name = path

        # Open image given specified path
        self.image = Image.open(path)

        # Get the width of image
        self.width = self.image.width

        # Get the height of image
        self.height = self.image.height

        # Create two-dimensional matrix for energy
        self.energy_matrix = [[0] * self.width for i in range(self.height)]

        # Create two-dimensional RGB matrix for image
        self.image_matrix = np.asarray(self.image).tolist()

    def update(self):

        # Get the width of image
        self.width = len(self.image_matrix[0])

        # Get the height of image
        self.height = len(self.image_matrix)

    def show_image(self):
        # Show image
        self.image.show()

    def get_image_matrix(self):
        # Return RGB matrix of image
        return self.image

    def get_energy_matrix(self):
        # Return computed energy matrix of image
        return self.energy_matrix

    def compute_delta_x(self, x, y):
        # Get the right neighbor of current pixel
        right = self.image_matrix[y][(x + 1) % self.width]

        # Get the left neighbor of current pixel
        left = self.image_matrix[y][(x - 1) % self.width]

        # Compute Delta-X
        return (right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2 + (right[2] - left[2]) ** 2

    def compute_delta_y(self, x, y):
        # Get the bottom neighbor of current pixel
        bottom = self.image_matrix[(y + 1) % self.height][x]

        # Get the top neighbor of current pixel
        top = self.image_matrix[(y - 1) % self.height][x]

        # Compute Delta-Y
        return (bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2 + (bottom[2] - top[2]) ** 2

    def compute_energy_matrix(self):
        # Coordinate System
        # (0,0) in the upper left corner.
        # Coordinates are usually passed to the library as 2-tuples (x, y).

        # Iterate over image
        for x in range(self.width):
            for y in range(self.height):

                # Get the corresponding value for Delta-X and Delta-Y
                delta_x = self.compute_delta_x(x, y)
                delta_y = self.compute_delta_y(x, y)

                # Compute energy
                energy = (delta_x + delta_y)**0.5

                self.energy_matrix[y][x] = energy

    def create_energy_image(self):
        # Create image from energy matrix
        self.energy_image = Image.fromarray(np.array(self.energy_matrix))

        # Convert to RGB mode
        self.energy_image = self.energy_image.convert("RGB")

        # Show the energy image
        # self.energy_image.show()

        # Store image in CWD
        # self.energy_image.save("energy_image.png")

    def plot_seam(self, path, iteration):
        self.create_energy_image()

        # Change the color of pixel in path
        for (x, y) in path:
            self.energy_image.putpixel((x, y), (255, 0, 0))

        # Show image
        # self.energy_image.show()

        # Save energy image
        self.energy_image.save(
            "energy_images/" + iteration + " manipulated_image.png")

    def remove_vertical_seam_from_energy(self, path):
        for (x, y) in path:
            self.energy_matrix[y].pop(x)

        return self.create_energy_image()

    def remove_horizontal_seam_from_energy(self, path):

        # Do some preprocessing in order to obtain transpose of new_image
        new_matrix = [[0] * self.height for i in range(self.width)]

        for i in range(len(self.energy_matrix)):
            for j in range(len(self.energy_matrix[i])):
                new_matrix[j][i] = self.energy_matrix[i][j]

        # Remove the pixels that are in the path
        for (x, y) in path:
            new_matrix[x].pop(y)

        w = len(new_matrix[0])
        h = len(new_matrix)

        self.energy_matrix = [[0] * h for i in range(w)]
        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[i])):
                self.energy_matrix[j][i] = new_matrix[i][j]

    def remove_vertical_seam(self, path):
        # Remove the pixels that are in the path
        for (x, y) in path:
            self.image_matrix[y].pop(x)

    def remove_horizontal_seam(self, path):

        # Do some preprocessing in order to obtain transpose of new_image
        new_matrix = [[0] * self.height for i in range(self.width)]

        for i in range(len(self.image_matrix)):
            for j in range(len(self.image_matrix[i])):
                new_matrix[j][i] = self.image_matrix[i][j]

        # Remove the pixels that are in the path
        for (x, y) in path:
            new_matrix[x].pop(y)

        w = len(new_matrix[0])
        h = len(new_matrix)

        self.image_matrix = [[0] * h for i in range(w)]

        for i in range(len(new_matrix)):
            for j in range(len(new_matrix[i])):
                self.image_matrix[j][i] = new_matrix[i][j]

    def create_resized_image(self):

        # Create new Image obj
        img = Image.new(self.image.mode, (self.width, self.height))


        w = self.width
        h = self.height
        
        
        # Copy remaining pixel from new_image into newly created image obj
        for x in range(w):
            for y in range(h):
                img.putpixel((x, y), tuple(self.image_matrix[y][x]))

        
        self.image = img

        self.image.save("resized_image.png")


class SeamFinding:
    def __init__(self):
        # Init the intial node
        self.start = "S"

        # Init a priority queue
        self.priority_q = []

    def intializer(self, energy_matrix):
        # Graph !
        self.graph = energy_matrix

        self.width = len(self.graph[0])
        self.height = len(self.graph)

        # Store distance of each node from source node
        self.distance = [[(math.inf, None)] *
                         self.width for i in range(self.height)]

    def get_successor(self, node):
        pass

    def get_cost(self, node):
        # Return cost of node
        (x, y) = node
        return self.graph[y][x]

    def compute_distance(self):
        # Set distance and parent for source node
        # self.distance[(self.start)] = (0, None)

        # Set distance to inf and parent to None for all nodes
        # for x in range(self.width):
        #     for y in range(self.height):
        #         self.distance[y][x] = (math.inf, None)

        # Add source node with corresponding distance to heap
        heappush(self.priority_q, (0, self.start))

        # Loop until pq gets free
        while len(self.priority_q) != 0:

            # Pop node with lowest value of distance
            (distance, v) = heappop(self.priority_q)

            # Iterate over successor of v
            for successor in self.get_successor(v):

                # Compute cost from source node to current successor
                alt = distance + self.get_cost(successor)

                # If we got better cost from source node to current successor,
                # Update distance with corresponding parent
                (x, y) = successor
                if alt < self.distance[y][x][0]:
                    self.distance[y][x] = (alt, v)

                    # Add successor to pq
                    heappush(self.priority_q, (alt, successor))

    def get_seam(self):
        pass

    def get_path(self, node, path):
        # Add current node to path
        path = path + [node]

        # Get the parent of curret node
        (x, y) = node
        (energy, parent) = self.distance[y][x]

        # While parent is not S, continue until reach S
        while parent != self.start:

            # Add seen node in path
            path = path + [parent]
            (x, y) = parent
            (_, parent) = self.distance[y][x]

        return path


class HorizontalSeamFinding(SeamFinding):
    def get_successor(self, node):
        # If the node is source node, return all nodes in first row as its successors
        successors = []
        if node == self.start:
            x = 0
            for y in range(self.height):
                successors.append((x, y))
            return successors

        # Add Left
        (x, y) = node
        if x < self.width - 1:
            successors.append((x + 1, y))

        # Up left
        if x < self.width - 1 and y > 0:
            successors.append((x + 1, y - 1))

        # Down left nodes
        if x < self.width - 1 and y < self.height - 1:
            successors.append((x + 1, y + 1))

        return successors

    def get_seam(self):
        # Set the index of last row
        x = self.width - 1

        # Declare best node var
        best_node = None

        #
        min_energy = math.inf

        # Find the node that has minimun total energy from source
        for y in range(len(self.distance)):
            (energy, node) = self.distance[y][x]
            if energy < min_energy:
                min_energy = energy
                best_node = (x, y)

        # Call a method to get the min cost path
        return self.get_path(best_node, [])


class VerticalSeamFinding(SeamFinding):

    def get_successor(self, node):
        # If the node is source node, return all nodes in first row as its successors
        successors = []
        if node == self.start:
            y = 0
            for x in range(self.width):
                successors.append((x, y))
            return successors

        # Add bottom
        (x, y) = node
        if y < self.height - 1:
            successors.append((x, y + 1))

        # Bottom left
        if y < self.height - 1 and x > 0:
            successors.append((x - 1, y + 1))

        # Bottom right nodes
        if y < self.height - 1 and x < self.width - 1:
            successors.append((x + 1, y + 1))

        return successors

    def get_seam(self):
        # Set the index of last row
        y = self.height - 1

        # Declare best node var
        best_node = None

        #
        min_energy = math.inf

        # Find the node that has minimun total energy from source
        for x in range(len(self.distance[y])):
            (energy, node) = self.distance[y][x]
            if energy < min_energy:
                min_energy = energy
                best_node = (x, y)

        # Call a method to get the min cost path
        return self.get_path(best_node, [])


def main(argv):
    input_file = ''
    vseam = ''
    hseam = ''
    try:
        opts, args = getopt.getopt(
            argv, "i:h:v:v", ["ifile=", "hseam=", "vseam="])
    except getopt.GetoptError:
        print('main.py -i <inputfile> -h <horizontalSeam> -v <verticalSeam>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-v", "--vseam"):
            vseam = arg
        elif opt in ("-h", "--hseam"):
            hseam = arg

    hseam = int(hseam)
    vseam = int(vseam)

    return calc(input_file, hseam, vseam)


def calc(input_file, h, v):

    # 
    if not os.path.exists('energy_images'):
        os.makedirs('energy_images')


    # create a seam carving instance
    seamCarving = SeamCarving(input_file)

    # Create an instance of SeamFinding class
    horizontalSeamFinding = HorizontalSeamFinding()
    verticalSeamFinding = VerticalSeamFinding()

    logging.info('Computing energy matrix')
    seamCarving.compute_energy_matrix()

    for i in range(h):
        logging.info("# Starting "+str(i)+"'th Iteration - Horizontal")

        logging.info('Getting energy matrix')
        graph = seamCarving.get_energy_matrix()

        logging.info('Dijkstra initialization')
        horizontalSeamFinding.intializer(graph)

        logging.info('Computing distance table')
        horizontalSeamFinding.compute_distance()

        logging.info('Finding minimum energy seam')
        path = horizontalSeamFinding.get_seam()

        logging.info('Creating energy image with minimum energy seam plotted')
        seamCarving.plot_seam(path, "Horizontal - " + str(i))

        logging.info('Removing found seam from image matrix')
        seamCarving.remove_horizontal_seam(path)

        logging.info('Removing found seam from energy matrix')
        seamCarving.remove_horizontal_seam_from_energy(path)

        seamCarving.update()

    logging.info('Computing energy matrix')
    seamCarving.compute_energy_matrix()

    for i in range(v):
        logging.info("# Starting "+str(i)+" Iteration - Vertical")

        logging.info('Getting energy matrix')
        graph = seamCarving.get_energy_matrix()

        logging.info('Dijkstra initialization')
        verticalSeamFinding.intializer(graph)

        logging.info('Computing distance table')
        verticalSeamFinding.compute_distance()

        logging.info('Finding minimum energy seam')
        path = verticalSeamFinding.get_seam()

        logging.info('Creating energy image with minimum energy seam plotted')
        seamCarving.plot_seam(path, "Vertical - " + str(i))

        logging.info('Removing found seam from image matrix')
        seamCarving.remove_vertical_seam(path)

        logging.info('Removing found seam from energy matrix')
        seamCarving.remove_vertical_seam_from_energy(path)

        seamCarving.update()


    seamCarving.create_resized_image()

if __name__ == "__main__":
    main(sys.argv[1:])
