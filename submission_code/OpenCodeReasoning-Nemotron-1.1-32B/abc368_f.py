import sys

MAX = 100000

def main():
	divisors = [[] for _ in range(MAX + 1)]
	for i in range(1, MAX + 1):
		for j in range(i, MAX + 1, i):
			divisors[j].append(i)
	
	g = [0] * (MAX + 1)
	for n in range(2, MAX + 1):
		s = set()
		for d in divisors[n]:
			if d == n:
				continue
			s.add(g[d])
		mex_val = 0
		while mex_val in s:
			mex_val += 1
		g[n] = mex_val

	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1 + n]))
	
	nim = 0
	for a in arr:
		nim ^= g[a]
		
	print("Anna" if nim != 0 else "Bruno")

if __name__ == '__main__':
	main()