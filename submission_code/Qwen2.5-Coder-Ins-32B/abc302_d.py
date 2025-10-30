# YOUR CODE HERE
import sys

def main():
    import sys
    import bisect

    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    D = int(input[2])
    A = list(map(int, input[3:3+N]))
    B = list(map(int, input[3+N:3+N+M]))

    A.sort()
    B.sort()

    max_sum = -1
    for a in A:
        idx = bisect.bisect_right(B, a + D)
        if idx > 0:
            b = B[idx - 1]
            if abs(a - b) <= D:
                max_sum = max(max_sum, a + b)

    print(max_sum)

if __name__ == "__main__":
    main()