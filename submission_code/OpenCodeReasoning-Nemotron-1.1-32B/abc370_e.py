mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	K = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	prefix_map = {0: 1}
	total_dp = 1
	current_prefix = 0
	
	for i in range(n):
		current_prefix += A[i]
		x = current_prefix - K
		subtract = prefix_map.get(x, 0)
		dp_i = (total_dp - subtract) % mod
		total_dp = (total_dp + dp_i) % mod
		prefix_map[current_prefix] = (prefix_map.get(current_prefix, 0) + dp_i) % mod
	
	print(dp_i % mod)

if __name__ == '__main__':
	main()