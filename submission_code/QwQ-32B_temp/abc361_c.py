import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    window_size = N - K
    min_diff = float('inf')
    for i in range(N - window_size + 1):
        current_diff = A[i + window_size - 1] - A[i]
        if current_diff < min_diff:
            min_diff = current_diff
    print(min_diff)

if __name__ == "__main__":
    main()