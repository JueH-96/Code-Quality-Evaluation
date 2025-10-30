import sys

MAX_VAL = 500000

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	contests = []
	index = 1
	for i in range(n):
		L = int(data[index])
		R = int(data[index + 1])
		index += 2
		contests.append((L, R))
	
	q = int(data[index])
	index += 1
	queries = []
	for i in range(q):
		X = int(data[index])
		index += 1
		queries.append(X)
	
	size = MAX_VAL + 2
	bit = [0] * size

	def update(i, delta):
		while i <= MAX_VAL:
			bit[i] += delta
			i += i & -i

	def query(i):
		s = 0
		while i:
			s += bit[i]
			i -= i & -i
		return s

	for L, R in contests:
		lo, hi = 1, MAX_VAL
		while lo < hi:
			mid = (lo + hi) // 2
			if mid + query(mid) >= L:
				hi = mid
			else:
				lo = mid + 1
		X_low = lo

		lo, hi = 1, MAX_VAL
		while lo < hi:
			mid = (lo + hi + 1) // 2
			if mid + query(mid) <= R:
				lo = mid
			else:
				hi = mid - 1
		X_high = lo

		if X_low <= X_high:
			update(X_low, 1)
			if X_high + 1 <= MAX_VAL:
				update(X_high + 1, -1)
	
	for X in queries:
		res = X + query(X)
		print(res)

if __name__ == '__main__':
	main()