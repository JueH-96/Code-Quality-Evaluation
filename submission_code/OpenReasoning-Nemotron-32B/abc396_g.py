import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	H, W = map(int, data[0].split())
	n = 1 << W
	freq = [0] * n
	for i in range(1, 1 + H):
		s = data[i].strip()
		num = int(s, 2)
		freq[num] += 1

	g = [0] * n
	for k in range(n):
		cnt = bin(k).count('1')
		g[k] = min(cnt, W - cnt)
	
	def FWT(a):
		m = len(a)
		d = 1
		while d < m:
			for i in range(0, m, 2 * d):
				for j in range(i, i + d):
					x = a[j]
					y = a[j + d]
					a[j] = x + y
					a[j + d] = x - y
			d <<= 1

	A = freq.copy()
	B = g.copy()
	FWT(A)
	FWT(B)
	
	C = [A[i] * B[i] for i in range(n)]
	FWT(C)
	conv = [C[i] // n for i in range(n)]
	
	ans = min(conv)
	print(ans)

if __name__ == '__main__':
	main()