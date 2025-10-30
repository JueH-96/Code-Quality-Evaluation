import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	total_vars = 2 * n
	equations = []
	occ = [set() for _ in range(total_vars)]
	
	index = 2
	for i in range(m):
		a = int(data[index])
		b = int(data[index+1])
		c = int(data[index+2])
		index += 3
		a0 = a - 1
		b0 = b - 1
		vars_set = {a0, b0, a0 + n}
		equations.append([set(vars_set), c])
		for v in vars_set:
			occ[v].add(i)
	
	sol = [None] * total_vars
	q = deque()
	
	for i in range(m):
		if len(equations[i][0]) == 1:
			q.append(i)
	
	while True:
		while q:
			i = q.popleft()
			vars_set, const = equations[i]
			if not vars_set:
				if const != 0:
					print(-1)
					return
				continue
			v = next(iter(vars_set))
			sol[v] = const
			if i in occ[v]:
				occ[v].remove(i)
			for j in list(occ[v]):
				if j == i:
					continue
				if v in equations[j][0]:
					equations[j][0].discard(v)
					equations[j][1] ^= sol[v]
					occ[v].discard(j)
					if len(equations[j][0]) == 1:
						q.append(j)
		
		found = False
		for v in range(total_vars):
			if sol[v] is None and occ[v]:
				found = True
				break
		if not found:
			break
			
		sol[v] = 0
		for j in list(occ[v]):
			if v in equations[j][0]:
				equations[j][0].discard(v)
				occ[v].discard(j)
				if len(equations[j][0]) == 1:
					q.append(j)
	
	for i in range(m):
		vars_set, const = equations[i]
		if not vars_set and const != 0:
			print(-1)
			return
	
	for i in range(total_vars):
		if sol[i] is None:
			sol[i] = 0
	
	s = ''.join('1' if sol[i + n] else '0' for i in range(n))
	print(s)

if __name__ == "__main__":
	main()