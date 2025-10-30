import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	
	if m == 0:
		print(0)
		return
		
	parent = list(range(n+1))
	rank = [0] * (n+1)
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		x = find(x)
		y = find(y)
		if x == y:
			return False
		if rank[x] < rank[y]:
			parent[x] = y
		elif rank[x] > rank[y]:
			parent[y] = x
		else:
			parent[y] = x
			rank[x] += 1
		return True

	components = n
	idx = 2
	for _ in range(m):
		u = int(data[idx])
		v = int(data[idx+1])
		idx += 2
		if union(u, v):
			components -= 1
			
	ans = m - n + components
	print(ans)

if __name__ == "__main__":
	main()