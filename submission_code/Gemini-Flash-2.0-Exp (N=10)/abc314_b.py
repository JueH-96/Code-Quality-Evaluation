def solve():
    n = int(input())
    bets = []
    for _ in range(n):
        c = int(input())
        a = list(map(int, input().split()))
        bets.append((c, a))
    x = int(input())

    winners = []
    for i in range(n):
        if x in bets[i][1]:
            winners.append((i + 1, bets[i][0]))
    
    if not winners:
        print(0)
        return

    min_bets = min(w[1] for w in winners)
    result = [w[0] for w in winners if w[1] == min_bets]
    result.sort()

    print(len(result))
    print(*result)

solve()