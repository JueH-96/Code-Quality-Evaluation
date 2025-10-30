import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total = [0] * (1 << n)
	for mask in range(1 << n):
		s_val = 0
		for i in range(n):
			if mask & (1 << i):
				s_val += A[i]
		total[mask] = s_val

	dp = [set() for _ in range(1 << n)]
	dp[0] = {0}
	
	for mask in range(1, 1 << n):
		dp[mask] = set()
		sub = mask
		while sub:
			s_val = total[sub]
			rest = mask ^ sub
			for x in dp[rest]:
				new_val = s_val ^ x
				dp[mask].add(new_val)
			sub = (sub - 1) & mask
			
	print(len(dp[(1 << n) - 1]))

if __name__ == '__main__':
	main()