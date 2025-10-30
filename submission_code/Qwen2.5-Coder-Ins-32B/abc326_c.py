# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    A = list(map(int, input[2:]))
    A.sort()

    max_gifts = 0
    for i in range(N):
        # Find the rightmost index j such that A[j] < A[i] + M
        j = bisect.bisect_left(A, A[i] + M)
        max_gifts = max(max_gifts, j - i)

    print(max_gifts)

if __name__ == "__main__":
    main()