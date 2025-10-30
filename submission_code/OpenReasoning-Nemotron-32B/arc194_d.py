mod = 998244353

def main():
	import sys
	from collections import deque
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	def decompose(l, r):
		res = []
		i = l
		while i <= r:
			cnt = 0
			j = i
			while j <= r:
				if s[j] == '(':
					cnt += 1
				else:
					cnt -= 1
				if cnt == 0:
					break
				j += 1
			if cnt != 0:
				break
			res.append((i, j))
			i = j + 1
		return res

	top_level = decompose(0, n - 1)
	ans = 1
	if len(top_level) >= 2:
		ans = (ans * 2) % mod
		
	q = deque(top_level)
	while q:
		l, r = q.popleft()
		if r - l + 1 == 2:
			continue
		children = decompose(l + 1, r - 1)
		c = len(children)
		if c >= 2:
			ans = (ans * 2) % mod
		for child in children:
			q.append(child)
			
	print(ans)

if __name__ == "__main__":
	main()