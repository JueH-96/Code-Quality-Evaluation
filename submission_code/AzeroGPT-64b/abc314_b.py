n = int(input())
people_bets = []

for _ in range(n):
    bets = int(input())
    bets_list = list(map(int, input().split()))
    people_bets.append(bets_list)

result = int(input())

min_len = 38
winners = []

for i in range(n):
    if result in people_bets[i]:
        if len(people_bets[i]) < min_len:
            min_len = len(people_bets[i])
            winners.clear()
            winners.append(i + 1)
        elif len(people_bets[i]) == min_len:
            winners.append(i + 1)

print(len(winners))
if len(winners) > 0:
    print(*winners)