def dfs(current, moves_done, memo):
	if moves_done == 10:
		return False
	if moves_done % 2 == 1:
		if current % 3 == 0:
			return True
	key = (current, moves_done)
	if key in memo:
		return memo[key]
	
	if moves_done % 2 == 0:
		for move in [1, -1]:
			next_val = current + move
			if dfs(next_val, moves_done + 1, memo):
				memo[key] = True
				return True
		memo[key] = False
		return False
	else:
		for move in [1, -1]:
			next_val = current + move
			if not dfs(next_val, moves_done + 1, memo):
				memo[key] = False
				return False
		memo[key] = True
		return True

t = int(input())
for _ in range(t):
	n = int(input())
	memo = {}
	res = dfs(n, 0, memo)
	print("First" if res else "Second")