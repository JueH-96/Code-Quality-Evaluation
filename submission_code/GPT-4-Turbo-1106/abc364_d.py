import sys

def main():
    # Read the number of points and queries
    N, Q = map(int, sys.stdin.readline().split())
    # Read the coordinates of points A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Process each query
    for _ in range(Q):
        b, k = map(int, sys.stdin.readline().split())
        # Calculate the distances and sort them
        distances = sorted(abs(a - b) for a in A)
        # Output the k-th smallest distance
        print(distances[k - 1])

if __name__ == "__main__":
    main()