n = int(input().strip())
persons = []
for i in range(n):
	c = int(input().strip())
	bets = list(map(int, input().split()))
	persons.append((i+1, c, set(bets)))
	
x = int(input().strip())

candidates = []
for (idx, c, bet_set) in persons:
	if x in bet_set:
		candidates.append((idx, c))

if not candidates:
	print(0)
else:
	min_bet = min(c for (_, c) in candidates)
	winners = [idx for (idx, c) in candidates if c == min_bet]
	winners.sort()
	print(len(winners))
	print(" ".join(map(str, winners)))