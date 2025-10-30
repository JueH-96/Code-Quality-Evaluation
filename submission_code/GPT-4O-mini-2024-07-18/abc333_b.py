# YOUR CODE HERE
import math

def distance(point1, point2):
    # Define the coordinates of the vertices of the pentagon
    coordinates = {
        'A': (1, 0),
        'B': (math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5)),
        'C': (math.cos(4 * math.pi / 5), math.sin(4 * math.pi / 5)),
        'D': (math.cos(6 * math.pi / 5), math.sin(6 * math.pi / 5)),
        'E': (math.cos(8 * math.pi / 5), math.sin(8 * math.pi / 5)),
    }
    
    x1, y1 = coordinates[point1]
    x2, y2 = coordinates[point2]
    
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    input_data = input().strip().split()
    S1, S2 = input_data[0][0], input_data[0][1]
    T1, T2 = input_data[1][0], input_data[1][1]
    
    length_S = distance(S1, S2)
    length_T = distance(T1, T2)
    
    if math.isclose(length_S, length_T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()