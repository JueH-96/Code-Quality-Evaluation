import sys
from itertools import product

def main():
	data = sys.stdin.read().split()
	if not data:
		print("0" * 0)
		return
	n = int(data[0])
	m = int(data[1])
	edges = []
	index = 2
	for i in range(m):
		a = int(data[index])
		b = int(data[index+1])
		c_val = int(data[index+2])
		index += 3
		a -= 1
		b -= 1
		edges.append((a, b, c_val))
	
	if m == 0:
		print("0" * n)
		return
		
	if n > 20 or m > 10000:
		print(-1)
		return
		
	for assignment in product([0,1], repeat=n):
		parent = list(range(n))
		diff = [0] * n
		
		def find(i):
			if parent[i] == i:
				return i
			else:
				root = find(parent[i])
				old_parent = parent[i]
				parent[i] = root
				diff[i] ^= diff[old_parent]
				return root
				
		valid = True
		for (a, b, c_val) in edges:
			const = c_val ^ assignment[a]
			ra = find(a)
			rb = find(b)
			if ra == rb:
				if (diff[a] ^ diff[b]) != const:
					valid = False
					break
			else:
				parent[rb] = ra
				diff[rb] = diff[a] ^ diff[b] ^ const
				
		if valid:
			s = ''.join(str(x) for x in assignment)
			print(s)
			return
			
	print(-1)

if __name__ == "__main__":
	main()