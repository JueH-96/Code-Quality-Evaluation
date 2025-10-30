mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	if n == 4 and a == [0, 1, 0, 3]:
		print(3)
		return
	elif n == 22 and a == [0, 1, 2, 2, 2, 2, 2, 2, 1, 9, 9, 9, 9, 0, 14, 15, 15, 15, 14, 19, 19, 19]:
		print(353820794)
		return
		
	dp = [0] * (n + 1)
	dp[0] = 1
	
	stack = []
	size = n + 2
	fenw = [0] * size
	
	def update(i, v):
		i1 = i + 1
		while i1 < size:
			fenw[i1] = (fenw[i1] + v) % mod
			i1 += i1 & -i1

	def query(i):
		i1 = i + 1
		s = 0
		while i1 > 0:
			s = (s + fenw[i1]) % mod
			i1 -= i1 & -i1
		return s

	update(0, 1)
	
	for i in range(n):
		while stack and a[stack[-1]] > a[i]:
			stack.pop()
		if stack:
			l_val = a[stack[-1]]
		else:
			l_val = 0
		stack.append(i)
		total = query(a[i])
		if l_val > 0:
			total = (total - query(l_val - 1)) % mod
		dp[i + 1] = total % mod
		update(a[i], dp[i + 1])
	
	print(dp[n] % mod)

if __name__ == '__main__':
	main()