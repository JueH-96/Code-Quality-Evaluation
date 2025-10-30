def solve():
    n, t = map(int, input().split())
    scores = [0] * n
    for _ in range(t):
        a, b = map(int, input().split())
        scores[a - 1] += b
        distinct_scores = set(scores)
        print(len(distinct_scores))

solve()