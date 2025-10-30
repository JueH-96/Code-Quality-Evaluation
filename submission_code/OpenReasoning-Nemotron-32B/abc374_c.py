import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	K = list(map(int, data[1:1+n]))
	total = sum(K)
	min_max = total
	dp = [0] * (1 << n)
	for mask in range(1, 1 << n):
		lowbit = mask & -mask
		j = lowbit.bit_length() - 1
		dp[mask] = dp[mask ^ lowbit] + K[j]
		current_max = max(dp[mask], total - dp[mask])
		if current_max < min_max:
			min_max = current_max
	print(min_max)

if __name__ == "__main__":
	main()