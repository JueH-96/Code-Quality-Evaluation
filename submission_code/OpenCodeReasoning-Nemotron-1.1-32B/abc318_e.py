def main():
	import sys
	from collections import defaultdict

	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	pos = defaultdict(list)
	for idx, num in enumerate(arr):
		pos[num].append(idx)
		
	ans = 0
	for indices in pos.values():
		m = len(indices)
		if m < 2:
			continue
		for j in range(m):
			term = (indices[j] - j) * (2 * j - (m - 1))
			ans += term
			
	print(ans)

if __name__ == "__main__":
	main()