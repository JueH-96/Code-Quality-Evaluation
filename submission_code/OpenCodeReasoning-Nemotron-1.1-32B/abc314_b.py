n = int(input().strip())
bets_list = []
for i in range(n):
	c = int(input().strip())
	arr = list(map(int, input().split()))
	s = set(arr)
	bets_list.append((i+1, c, s))

x = int(input().strip())

candidates = []
for (idx, c, s) in bets_list:
	if x in s:
		candidates.append((idx, c))

if not candidates:
	print(0)
else:
	min_bet = min(c for idx, c in candidates)
	result = [idx for idx, c in candidates if c == min_bet]
	result.sort()
	print(len(result))
	print(" ".join(map(str, result)))