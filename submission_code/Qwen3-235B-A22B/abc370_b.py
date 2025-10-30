import sys

def main():
    n = int(sys.stdin.readline())
    A = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        values = list(map(int, sys.stdin.readline().split()))
        for j in range(1, i + 1):
            A[i][j] = values[j - 1]
    current = 1
    for j in range(1, n + 1):
        if current >= j:
            current = A[current][j]
        else:
            current = A[j][current]
    print(current)

if __name__ == "__main__":
    main()