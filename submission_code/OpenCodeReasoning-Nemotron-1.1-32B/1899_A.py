import sys

def solve(n, moves_done, memo):
	if moves_done == 10:
		return 'Vova'
	key = (n, moves_done)
	if key in memo:
		return memo[key]
	
	if moves_done % 2 == 0:
		if (n + 1) % 3 == 0 or (n - 1) % 3 == 0:
			memo[key] = 'Vanya'
			return 'Vanya'
		else:
			next1 = solve(n + 1, moves_done + 1, memo)
			next2 = solve(n - 1, moves_done + 1, memo)
			if next1 == 'Vanya' or next2 == 'Vanya':
				memo[key] = 'Vanya'
				return 'Vanya'
			else:
				memo[key] = 'Vova'
				return 'Vova'
	else:
		next1 = solve(n + 1, moves_done + 1, memo)
		next2 = solve(n - 1, moves_done + 1, memo)
		if next1 == 'Vova' or next2 == 'Vova':
			memo[key] = 'Vova'
			return 'Vova'
		else:
			memo[key] = 'Vanya'
			return 'Vanya'

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	n_list = list(map(int, data[1:1 + t]))
	memo = {}
	results = []
	for n_val in n_list:
		res = solve(n_val, 0, memo)
		results.append("First" if res == 'Vanya' else "Second")
	for res in results:
		print(res)

if __name__ == "__main__":
	main()