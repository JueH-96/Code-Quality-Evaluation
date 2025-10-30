def main():
    import sys
    import threading
    def solve():
        import sys
        from bisect import bisect_right, bisect_left

        data = sys.stdin.read().split()
        N = int(data[0])
        M = int(data[1])
        A = list(map(int, data[2:2+N]))
        B = list(map(int, data[2+N:2+N+M]))
        A.sort()
        B.sort()

        # f(X) = (# sellers with A_i <= X) - (# buyers with B_i >= X)
        # We want the smallest X such that f(X) >= 0.
        # f is non-decreasing, so we can binary search.
        low = 0
        high = 10**9 + 1  # high is always a valid point: f(high) = N - 0 >= 0

        while high - low > 1:
            mid = (low + high) // 2
            sellers = bisect_right(A, mid)
            buyers = M - bisect_left(B, mid)
            if sellers >= buyers:
                high = mid
            else:
                low = mid

        print(high)

    if sys.stdin.isatty():
        # Running in an interactive environment
        solve()
    else:
        # On an online judge
        threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()