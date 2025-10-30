# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A = [0] + A  # 1-based indexing
    pos = [0] * (N + 1)
    for i in range(1, N+1):
        pos[A[i]] = i
    swaps = []
    for i in range(1, N+1):
        if A[i] != i:
            j = pos[i]
            val = A[i]
            A[i], A[j] = A[j], A[i]
            swaps.append((i, j))
            pos[i] = i
            pos[val] = j
    print(len(swaps))
    for swap in swaps:
        print(swap[0], swap[1])

if __name__ == "__main__":
    main()