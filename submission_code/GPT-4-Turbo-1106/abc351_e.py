import sys

def calculate_distance(A, B):
    # The rabbit can only reach a point B from point A if the parity of the sum of coordinates for A and B is the same.
    if (A[0] + A[1]) % 2 != (B[0] + B[1]) % 2:
        return 0
    # The distance is the maximum of the absolute differences of the coordinates.
    return max(abs(A[0] - B[0]), abs(A[1] - B[1]))

def main():
    # Read the number of points N
    N = int(sys.stdin.readline().strip())
    # Read the points coordinates
    points = [tuple(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    
    # Calculate the sum of distances
    total_distance = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            total_distance += calculate_distance(points[i], points[j])
    
    # Output the result
    print(total_distance)

if __name__ == "__main__":
    main()