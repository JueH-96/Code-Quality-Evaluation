from itertools import combinations

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    min_diff = float('inf')

    for indices in combinations(range(n), n - k):
        subsequence_b = [a[i] for i in sorted(indices)]
        if not subsequence_b:
            continue
        min_b = min(subsequence_b)
        max_b = max(subsequence_b)
        min_diff = min(min_diff, max_b - min_b)

    print(min_diff)

solve()