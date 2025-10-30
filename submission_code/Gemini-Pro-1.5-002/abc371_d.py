# YOUR CODE HERE
def solve():
    n = int(input())
    x = list(map(int, input().split()))
    p = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    for l, r in queries:
        total_villagers = 0
        for i in range(n):
            if l <= x[i] <= r:
                total_villagers += p[i]
        print(total_villagers)

solve()