def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # The last K cards are moved to the top
    B = A[N-K:] + A[:N-K]

    print(" ".join(map(str, B)))

# Call solve() function
if __name__ == "__main__":
    solve()