import sys
import threading

def main():
    import sys
    import bisect

    input = sys.stdin.readline
    N, Q = map(int, input().split())
    A = list(map(int, input().split()))

    # Bucket queries by their R value
    queries_by_R = [[] for _ in range(N + 1)]
    for qi in range(Q):
        R, X = map(int, input().split())
        queries_by_R[R].append((X, qi))

    dp = []       # dp[k] = minimal tail value of an increasing subsequence of length k+1
    ans = [0] * Q

    # Build dp incrementally for prefixes 1..N
    for i in range(1, N + 1):
        a = A[i - 1]
        # Standard LIS update (patience sorting idea)
        pos = bisect.bisect_left(dp, a)
        if pos == len(dp):
            dp.append(a)
        else:
            dp[pos] = a

        # Answer all queries with this R = i
        for X, qi in queries_by_R[i]:
            # We want the largest k such that dp[k-1] <= X.
            # bisect_right(dp, X) gives count of dp-elements <= X,
            # which is exactly the LIS length with cap X.
            ans[qi] = bisect.bisect_right(dp, X)

    # Output answers in original order
    sys.stdout.write("
".join(map(str, ans)))

if __name__ == "__main__":
    main()