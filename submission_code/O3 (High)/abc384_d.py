import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, S = data[0], data[1]
    A = data[2: 2 + N]

    total = sum(A)            # sum of one whole period

    # Special case : period length is 1
    if N == 1:
        print("Yes" if S % A[0] == 0 else "No")
        return

    # If one whole period already gives S, we can take that subsequence
    if total == S:
        print("Yes")
        return

    # ------------------------------------------------------------
    # Case 1 : the sum of one period is larger than needed.
    # Then any valid subsequence is contained in at most one period
    # (length ≤ N).  We only have to examine at most 2·N elements.
    # ------------------------------------------------------------
    if total > S:
        i = j = 0          # window is [i , j)
        cur = 0            # current window sum
        while i < N:
            # extend window while it is short enough and total sum < S
            while j - i < N and cur < S:
                cur += A[j % N]
                j += 1
            if cur == S:
                print("Yes")
                return
            # shrink from the left
            cur -= A[i % N]
            i += 1
        print("No")
        return

    # ------------------------------------------------------------
    # Case 2 : total < S
    # We use a two–pointer window but jump entire periods at once.
    # ------------------------------------------------------------
    l = r = 0          # window [l , r)
    cur = 0
    while l < N:
        # expand until current sum reaches or exceeds S
        while cur < S:
            deficit = S - cur
            if deficit >= total:
                # add k full periods in one shot
                k = deficit // total          # k ≥ 1
                cur += k * total
                r  += k * N
            else:
                # add one single element
                cur += A[r % N]
                r  += 1
        if cur == S:
            print("Yes")
            return
        # shrink window from the left
        cur -= A[l % N]
        l  += 1

    print("No")


if __name__ == "__main__":
    main()