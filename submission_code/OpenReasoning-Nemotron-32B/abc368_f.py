import sys

def main():
	max_val = 100000
	divisors = [[] for _ in range(max_val + 1)]
	for i in range(1, max_val + 1):
		j = i * 2
		while j <= max_val:
			divisors[j].append(i)
			j += i

	g = [0] * (max_val + 1)
	g[1] = 0
	for n in range(2, max_val + 1):
		s = set()
		for d in divisors[n]:
			s.add(g[d])
		mex_val = 0
		while mex_val in s:
			mex_val += 1
		g[n] = mex_val

	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1 + n]))
	xor_val = 0
	for a in arr:
		xor_val ^= g[a]
	print("Anna" if xor_val != 0 else "Bruno")

if __name__ == "__main__":
	main()