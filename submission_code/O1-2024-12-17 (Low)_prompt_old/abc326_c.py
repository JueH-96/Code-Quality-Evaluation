def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # Sort the gift coordinates
    A.sort()

    # We will use a two-pointer approach to find the maximum number of gifts
    # that can be covered by a half-open interval of length M.
    j = 0
    max_gifts = 0

    for i in range(N):
        # Move j as far right as possible while A[j] < A[i] + M
        while j < N and A[j] < A[i] + M:
            j += 1
        # The number of gifts in the interval [A[i], A[i] + M) is j - i
        max_gifts = max(max_gifts, j - i)

    print(max_gifts)

def main():
    solve()

if __name__ == "__main__":
    main()