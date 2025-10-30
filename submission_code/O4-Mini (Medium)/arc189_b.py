def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    X = list(map(int, input().split()))
    # Compute consecutive differences y[i] = X[i+1] - X[i], i = 0..N-2
    y = [X[i+1] - X[i] for i in range(N-1)]
    # Split into odd‐index and even‐index diffs (1‐based indexing of y):
    # here y[0] is y_1 (odd), y[1] is y_2 (even), etc.
    odd = []
    even = []
    for i, d in enumerate(y):
        if (i & 1) == 0:
            odd.append(d)
        else:
            even.append(d)
    # Sort each non‐decreasing
    odd.sort()
    even.sort()
    # Merge back
    oi = ei = 0
    new_y = [0] * (N-1)
    for i in range(N-1):
        if (i & 1) == 0:
            new_y[i] = odd[oi]
            oi += 1
        else:
            new_y[i] = even[ei]
            ei += 1
    # Reconstruct X' from X'[0] = X[0], X'[i] = X'[i-1] + new_y[i-1]
    s = X[0]
    cur = X[0]
    for d in new_y:
        cur += d
        s += cur
    # print result
    print(s)

if __name__ == "__main__":
    main()