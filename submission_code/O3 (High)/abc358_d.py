import sys

def main() -> None:
    input = sys.stdin.readline

    # read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # sort both lists in non-decreasing order
    A.sort()
    B.sort()

    j = 0           # current position in A (boxes)
    total = 0       # total amount of money to pay

    for b in B:     # process requirements from the smallest to the largest
        # skip every box that is too small for this requirement
        while j < N and A[j] < b:
            j += 1

        # if there is no suitable box left, it is impossible
        if j == N:
            print(-1)
            return

        # use the smallest still-unused box that satisfies the requirement
        total += A[j]
        j += 1      # mark this box as used

    print(total)


if __name__ == "__main__":
    main()