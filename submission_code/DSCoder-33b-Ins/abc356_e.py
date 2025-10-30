# YOUR CODE HERE
import sys

def solve(N, A):
    total = 0
    for i in range(N):
        for j in range(i+1, N):
            total += max(A[i], A[j]) // min(A[i], A[j])
    return total

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    print(solve(N, A))

if __name__ == "__main__":
    main()