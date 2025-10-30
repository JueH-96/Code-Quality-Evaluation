def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    min_diff = float('inf')

    for i in range(1 << n):
        indices = []
        for j in range(n):
            if (i >> j) & 1:
                indices.append(j)

        if len(indices) == k:
            is_good = True
            subsequence = [p[idx] for idx in indices]
            subsequence.sort()

            if len(set(subsequence)) != k:
                continue

            first_val = subsequence[0]
            for l in range(1, k):
                if subsequence[l] != first_val + l:
                    is_good = False
                    break

            if is_good:
                diff = indices[-1] - indices[0]
                min_diff = min(min_diff, diff)

    print(min_diff)

solve()