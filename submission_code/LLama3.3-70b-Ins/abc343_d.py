import sys

def main():
    N, T = map(int, sys.stdin.readline().split())
    A = [0] * (N + 1)
    for _ in range(T):
        a, b = map(int, sys.stdin.readline().split())
        A[a] += b
        unique_values = set()
        for i in range(1, N + 1):
            unique_values.add(A[i])
        print(len(unique_values))

if __name__ == "__main__":
    main()