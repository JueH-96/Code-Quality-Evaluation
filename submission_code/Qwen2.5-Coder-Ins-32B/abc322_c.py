# YOUR CODE HERE
import sys

def main():
    import sys
    import bisect

    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:]))

    result = [0] * N
    for i in range(N):
        idx = bisect.bisect_left(A, i + 1)
        result[i] = A[idx] - (i + 1)

    for r in result:
        print(r)

if __name__ == "__main__":
    main()