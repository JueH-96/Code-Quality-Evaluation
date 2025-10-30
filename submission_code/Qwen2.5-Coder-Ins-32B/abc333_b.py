# YOUR CODE HERE
import sys

def distance(p1, p2):
    # Define the coordinates of the pentagon vertices
    coords = {
        'A': (1, 0),
        'B': (0.30901699437494745, 0.9510565162951535),
        'C': (-0.8090169943749473, 0.5877852522924731),
        'D': (-0.8090169943749473, -0.5877852522924731),
        'E': (0.30901699437494745, -0.9510565162951535)
    }
    # Calculate the Euclidean distance between two points
    return ((coords[p1][0] - coords[p2][0]) ** 2 + (coords[p1][1] - coords[p2][1]) ** 2) ** 0.5

def main():
    input = sys.stdin.read().strip().split()
    S1, S2 = input[0]
    T1, T2 = input[1]
    
    if distance(S1, S2) == distance(T1, T2):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()