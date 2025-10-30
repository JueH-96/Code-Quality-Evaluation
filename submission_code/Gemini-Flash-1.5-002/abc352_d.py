# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    p = list(map(int, input().split()))

    min_diff = float('inf')

    for i in range(1 << n):
        indices = []
        values = []
        for j in range(n):
            if (i >> j) & 1:
                indices.append(j + 1)
                values.append(p[j])

        if len(indices) != k:
            continue

        values.sort()
        is_consecutive = True
        for j in range(1, k):
            if values[j] != values[j-1] + 1:
                is_consecutive = False
                break
        
        if is_consecutive:
            min_diff = min(min_diff, indices[-1] - indices[0])

    print(min_diff)

solve()