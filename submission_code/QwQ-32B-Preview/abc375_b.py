import sys
import numpy as np

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    points = np.array([tuple(map(int, data[1:2*N+1]))], dtype=int).reshape(N, 2)
    
    # Calculate distances between consecutive points
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    
    # Distance from start to first point
    start_to_first = np.sqrt(points[0, 0]**2 + points[0, 1]**2)
    
    # Distance from last point to origin
    last_to_end = np.sqrt(points[-1, 0]**2 + points[-1, 1]**2)
    
    # Total cost
    total_cost = np.sum(distances) + start_to_first + last_to_end
    print(total_cost)

if __name__ == "__main__":
    main()