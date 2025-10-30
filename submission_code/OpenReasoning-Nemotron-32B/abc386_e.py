import sys
from itertools import combinations

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	if k == 1:
		print(max(A))
	else:
		k0 = min(k, n - k)
		if k0 == k:
			best = 0
			for comb in combinations(A, k):
				xor_val = 0
				for num in comb:
					xor_val ^= num
				if xor_val > best:
					best = xor_val
			print(best)
		else:
			total_xor = 0
			for a in A:
				total_xor ^= a
			best = 0
			for comb in combinations(A, n - k):
				xor_val = 0
				for num in comb:
					xor_val ^= num
				candidate = total_xor ^ xor_val
				if candidate > best:
					best = candidate
			print(best)

if __name__ == "__main__":
	main()