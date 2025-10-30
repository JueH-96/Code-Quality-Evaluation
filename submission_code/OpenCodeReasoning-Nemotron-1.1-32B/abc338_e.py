import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	total = 2 * n
	match = [0] * (total + 1)
	
	index = 1
	for _ in range(n):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		match[a] = b
		match[b] = a
		
	tree = [0] * (total + 1)
	
	def update(i, delta):
		while i <= total:
			tree[i] += delta
			i += i & -i
			
	def query(i):
		s = 0
		while i:
			s += tree[i]
			i -= i & -i
		return s
		
	def range_query(l, r):
		if l > r:
			return 0
		return query(r) - query(l - 1)
	
	for i in range(1, total + 1):
		if i < match[i]:
			j = match[i]
			if range_query(i + 1, j - 1) > 0:
				print("Yes")
				return
			update(j, 1)
		else:
			update(i, -1)
			
	print("No")

if __name__ == "__main__":
	main()