import sys

def find_difference(N, grid_A, grid_B):
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                return i + 1, j + 1

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    grid_A = data[1:N+1]
    grid_B = data[N+1:2*N+1]

    i, j = find_difference(N, grid_A, grid_B)
    print(i, j)

if __name__ == "__main__":
    main()