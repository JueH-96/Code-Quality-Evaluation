import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	n, m = map(int, data[0].split())
	masks = []
	for i in range(1, 1 + n):
		s = data[i].strip()
		mask_val = 0
		for j, char in enumerate(s):
			if char == 'o':
				mask_val |= (1 << j)
		masks.append(mask_val)
	
	total_mask = (1 << m) - 1
	dp = [10**9] * (1 << m)
	dp[0] = 0
	
	for mask_i in masks:
		for state in range(1 << m):
			if dp[state] != 10**9:
				new_state = state | mask_i
				if dp[state] + 1 < dp[new_state]:
					dp[new_state] = dp[state] + 1
	
	print(dp[total_mask])

if __name__ == "__main__":
	main()