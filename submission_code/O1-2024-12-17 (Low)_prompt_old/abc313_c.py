def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # If N = 1, no operations are required because
    # difference between min and max (only one element) is already 0
    if N == 1:
        print(0)
        return

    A.sort()
    S = sum(A)
    x = S // N        # The base value
    k = S % N         # How many elements must become x+1

    # We want exactly k largest elements to be x+1,
    # and the remaining (N-k) elements to be x.
    #
    # The cost in terms of "operations" is exactly the total increments needed
    # (which will also match the total decrements).
    #
    # For the first (N-k) elements, final value = x:
    #   increments = sum( max(0, x - A[i]) )
    # For the last k elements, final value = x+1:
    #   increments = sum( max(0, (x+1) - A[i]) )

    ans = 0
    # First (N-k) elements → x
    for i in range(N - k):
        if A[i] < x:
            ans += (x - A[i])
    # Last k elements → x+1
    for i in range(N - k, N):
        if A[i] < x + 1:
            ans += ((x + 1) - A[i])

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()