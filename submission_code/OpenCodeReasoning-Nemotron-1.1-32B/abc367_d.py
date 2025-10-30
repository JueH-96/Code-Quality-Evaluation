import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	P = [0] * n
	for i in range(1, n):
		P[i] = P[i-1] + A[i-1]
	
	total_sum = P[-1] + A[-1]
	R = total_sum % m
	
	residues = [p % m for p in P]
	
	freq_nonwrap = defaultdict(int)
	for r in residues:
		freq_nonwrap[r] += 1
		
	nonwrap = 0
	for count in freq_nonwrap.values():
		nonwrap += count * (count - 1) // 2
		
	wrap = 0
	freq_wrap = defaultdict(int)
	for i in range(n):
		r = residues[i]
		target = (r - R) % m
		wrap += freq_wrap.get(target, 0)
		freq_wrap[r] += 1
		
	print(nonwrap + wrap)

if __name__ == "__main__":
	main()