# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    ab = []
    for _ in range(n):
        ab.append(list(map(int, input().split())))

    ans = 0
    import itertools

    for p in itertools.permutations(range(n), k):
        val = 1
        for i in range(k):
            val = ab[p[i]][0] * val + ab[p[i]][1]
        ans = max(ans, val)
    
    print(ans)

solve()