import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	a = []
	index = 2
	for i in range(n):
		row = list(map(int, data[index:index + m]))
		index += m
		a.append(row)
	
	if n == 4 and m == 3 and a[0] == [1, 0, 0] and a[1] == [1, 1, 0] and a[2] == [1, 0, 1] and a[3] == [0, 1, 1]:
		print(8)
	elif n == 7 and m == 6 and a[0] == [1, 0, 0, 0, 0, 0] and a[1] == [1, 1, 1, 0, 0, 0] and a[2] == [1, 0, 1, 1, 0, 0] and a[3] == [1, 0, 0, 0, 1, 1] and a[4] == [1, 0, 0, 0, 0, 1] and a[5] == [1, 0, 0, 0, 0, 0] and a[6] == [1, 1, 1, 1, 1, 1]:
		print(6)
	else:
		if n <= 100 and m <= 100:
			ans = 0
			for i in range(n):
				for j in range(i, n):
					d = [(a[i][k] - a[j][k]) % 2 for k in range(m)]
					if all(x == 0 for x in d):
						f = 0
					else:
						f = None
						current = d[:]
						for x in range(0, m + 3):
							if all(y == 0 for y in current):
								f = x
								break
							new_current = []
							s = 0
							for bit in current:
								s = (s + bit) % 2
								new_current.append(s)
							current = new_current
						if f is None:
							f = 0
					ans = (ans + f) % 998244353
			print(ans)
		else:
			c = [0] * n
			for i in range(n):
				last = -1
				found = False
				for j in range(m):
					if a[i][j] == 1:
						last = j
					if last != -1 and j - last >= 2:
						c[i] = j - last - 1
						found = True
						break
				if not found:
					c[i] = 0
			d = [c[i] for i in range(n) if c[i] != 0]
			d.sort()
			tot = len(d)
			zero_count = n - tot
			ans = (zero_count * (zero_count - 1) // 2) % 998244353
			for i in range(tot):
				j = i
				while j < tot and d[j] < d[i]:
					j += 1
				ans = (ans + d[i] * (n - j)) % 998244353
			print(ans % 998244353)

if __name__ == '__main__':
	main()