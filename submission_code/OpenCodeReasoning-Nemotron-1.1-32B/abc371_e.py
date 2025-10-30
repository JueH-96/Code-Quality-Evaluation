import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	d = defaultdict(list)
	for i, num in enumerate(arr):
		d[num].append(i)
		
	total = 0
	for key, lst in d.items():
		m = len(lst)
		for j in range(m):
			idx = lst[j]
			if j < m-1:
				nxt = lst[j+1]
			else:
				nxt = n
			total += (idx + 1) * (nxt - idx)
			
	print(total)

if __name__ == "__main__":
	main()