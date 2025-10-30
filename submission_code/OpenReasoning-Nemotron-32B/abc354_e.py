def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	cards = []
	for i in range(1, 1 + n):
		a, b = map(int, data[i].split())
		cards.append((a, b))
	
	total_states = 1 << n
	dp = [False] * total_states
	
	for state in range(total_states):
		indices = []
		for i in range(n):
			if state & (1 << i):
				indices.append(i)
		k = len(indices)
		if k < 2:
			dp[state] = False
			continue
		
		found_win = False
		for i in range(k):
			for j in range(i + 1, k):
				idx1 = indices[i]
				idx2 = indices[j]
				a1, b1 = cards[idx1]
				a2, b2 = cards[idx2]
				if a1 == a2 or b1 == b2:
					new_state = state ^ (1 << idx1) ^ (1 << idx2)
					if not dp[new_state]:
						found_win = True
						break
			if found_win:
				break
		dp[state] = found_win
	
	if dp[total_states - 1]:
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == "__main__":
	main()