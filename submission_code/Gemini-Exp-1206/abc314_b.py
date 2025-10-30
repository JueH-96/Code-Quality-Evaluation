def solve():
    n = int(input())
    bets = []
    for _ in range(n):
        c = int(input())
        a = list(map(int, input().split()))
        bets.append((c, a))
    x = int(input())

    winners = []
    min_bets = float('inf')

    for i in range(n):
        if x in bets[i][1]:
            winners.append((bets[i][0], i + 1))
            min_bets = min(min_bets, bets[i][0])

    result = []
    for num_bets, person_id in winners:
        if num_bets == min_bets:
            result.append(person_id)

    result.sort()
    print(len(result))
    if result:
        print(*result)

solve()