import sys
from collections import defaultdict

def main():
	s = sys.stdin.readline().strip()
	d = defaultdict(list)
	for idx, char in enumerate(s):
		d[char].append(idx)
	
	ans = 0
	for char, lst in d.items():
		m = len(lst)
		if m < 2:
			continue
		total = sum(lst)
		S1 = 0
		for j in range(m):
			S1 += j * lst[j]
		total_diff = 2 * S1 - (m - 1) * total
		num_pairs = m * (m - 1) // 2
		T_c = total_diff - num_pairs
		ans += T_c
		
	print(ans)

if __name__ == "__main__":
	main()