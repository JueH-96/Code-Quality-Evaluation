def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, Q = map(int, data[:2])
    R = list(map(int, data[2:2+N]))
    queries = list(map(int, data[2+N:]))

    # Sort the reindeer requirements
    R.sort()

    # Create a prefix sums array
    prefix_sum = [0] * (N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + R[i]

    import bisect

    # For each query, find the maximum number of sleighs we can pull
    # using binary search on the prefix sums
    idx = 0
    answers = []
    for _ in range(Q):
        X = queries[idx]
        idx += 1
        # We need to find the largest k such that prefix_sum[k] <= X
        k = bisect.bisect_right(prefix_sum, X) - 1
        answers.append(str(k))

    print("
".join(answers))

def main():
    solve()

if __name__ == '__main__':
    main()