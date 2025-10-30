def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    counts = [2] * n
    for x in a:
        counts[x - 1] -= 1

    remaining = []
    for i in range(n):
        remaining.extend([i + 1] * counts[i])

    remaining.sort()
    
    total_weirdness = 0
    for i in range(0, len(remaining) - 1, 2):
        total_weirdness += remaining[i+1] - remaining[i]

    print(total_weirdness)

solve()