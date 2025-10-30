# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    A = []
    for x in range(1, N+1):
        for y in range(1, N+1):
            row = list(map(int, data[idx:idx+N]))
            A.append(row)
            idx += N
    # Flatten the 3D array into a 1D list for easier handling
    # Since A is already in the order x, y, z
    # We can compute the prefix sum in 3D
    # Initialize prefix sum array
    prefix = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                # Calculate the index in the flattened list
                # Since A is stored as x, y, z
                # The index is (x-1)*N*N + (y-1)*N + (z-1)
                # But since A is a list of lists, we need to access A[(x-1)*N + (y-1)][z-1]
                # Wait, no. A is a list of lists, where each x has N y's, and each y has N z's
                # So A is a list of N*N lists, each of size N
                # So to get A[x][y][z], we need to access A[(x-1)*N + (y-1)][z-1]
                # So the value is A[(x-1)*N + (y-1)][z-1]
                value = A[(x-1)*N + (y-1)][z-1]
                prefix[x][y][z] = prefix[x-1][y][z] + prefix[x][y-1][z] + prefix[x][y][z-1] - prefix[x-1][y-1][z] - prefix[x-1][y][z-1] - prefix[x][y-1][z-1] + prefix[x-1][y-1][z-1] + value
    Q = int(data[idx])
    idx += 1
    for _ in range(Q):
        Lx = int(data[idx])
        Rx = int(data[idx+1])
        Ly = int(data[idx+2])
        Ry = int(data[idx+3])
        Lz = int(data[idx+4])
        Rz = int(data[idx+5])
        idx += 6
        # Calculate the sum using the prefix sum array
        total = prefix[Rx][Ry][Rz] - prefix[Lx-1][Ry][Rz] - prefix[Rx][Ly-1][Rz] - prefix[Rx][Ry][Lz-1] + prefix[Lx-1][Ly-1][Rz] + prefix[Lx-1][Ry][Lz-1] + prefix[Rx][Ly-1][Lz-1] - prefix[Lx-1][Ly-1][Lz-1]
        print(total)

if __name__ == "__main__":
    main()