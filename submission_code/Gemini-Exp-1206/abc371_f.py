def solve():
    n = int(input())
    x = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    total_moves = 0
    for t, g in queries:
        t -= 1
        moves = 0
        diff = g - x[t]
        if diff > 0:
            for i in range(t, n - 1):
                if x[i+1] - x[i] > diff:
                    moves += diff * (i - t + 1)
                    for j in range(t, i + 1):
                        x[j] += diff
                    diff = 0
                    break
                else:
                    moves += (x[i+1] - x[i]) * (i - t + 1)
                    for j in range(t, i + 1):
                        x[j] = x[i+1]
                    diff -= (x[i+1] - x[i])
            if diff > 0:
                moves += diff * (n - t)
                for j in range(t, n):
                    x[j] += diff
        elif diff < 0:
            diff = -diff
            for i in range(t, 0, -1):
                if x[i] - x[i-1] > diff:
                    moves += diff * (t - i + 1)
                    for j in range(i, t + 1):
                        x[j] -= diff
                    diff = 0
                    break
                else:
                    moves += (x[i] - x[i-1]) * (t - i + 1)
                    for j in range(i, t + 1):
                        x[j] = x[i-1]
                    diff -= (x[i] - x[i-1])
            if diff > 0:
                moves += diff * (t + 1)
                for j in range(0, t + 1):
                    x[j] -= diff
        
        total_moves += moves

    print(total_moves)

solve()