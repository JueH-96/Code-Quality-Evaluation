def main():
    import sys
    from bisect import bisect_right
    input = sys.stdin.readline

    # read N (number of sleighs) and Q (number of queries)
    n, q = map(int, input().split())
    # read the reindeer requirements for each sleigh
    R = list(map(int, input().split()))

    # sort requirements and build prefix sums
    R.sort()
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + R[i]

    # answer each query by binary searching the prefix array
    for _ in range(q):
        X = int(input())
        # number of sleighs = largest k such that prefix[k] <= X
        ans = bisect_right(prefix, X) - 1
        print(ans)

if __name__ == "__main__":
    main()