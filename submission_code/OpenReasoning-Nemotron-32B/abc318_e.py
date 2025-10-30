import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	groups = defaultdict(list)
	for i, a in enumerate(A):
		groups[a].append(i)
	
	ans = 0
	for key, L in groups.items():
		m = len(L)
		if m < 2:
			continue
			
		S1 = 0
		S2 = 0
		for idx, pos in enumerate(L):
			S1 += pos * idx
			S2 += pos * (m - 1 - idx)
			
		T1 = m * (m - 1) // 2
		T2 = (m - 1) * m * (2 * m - 1) // 6
		
		total_x = S1 - S2 + (m - 1) * T1 - 2 * T2
		ans += total_x
		
	print(ans)

if __name__ == "__main__":
	main()