import sys
from collections import Counter

def main():
	data = sys.stdin.read().splitlines()
	t = int(data[0].strip())
	index = 1
	out_lines = []
	for _ in range(t):
		n, k = map(int, data[index].split())
		index += 1
		s = data[index].strip()
		index += 1
		m = n - k
		
		freq = Counter(s)
		T0 = 0
		for count in freq.values():
			if count % 2 == 0:
				T0 += count
			else:
				T0 += count - 1
		
		if m % 2 == 0:
			if m <= T0:
				out_lines.append("YES")
			else:
				out_lines.append("NO")
		else:
			if m <= T0 + 1:
				out_lines.append("YES")
			else:
				out_lines.append("NO")
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()