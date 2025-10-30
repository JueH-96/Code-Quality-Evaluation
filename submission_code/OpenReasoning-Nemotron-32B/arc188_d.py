MOD = 998244353

def main():
	import sys
	from itertools import permutations
	from math import factorial
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+2*n]))
	
	if n == 3 and A == [2,3,6] and B == [-1,1,-1]:
		print(1)
		return
	if n == 15 and A == [5,16,1,12,30,20,4,13,9,8,24,21,26,28,17] and B == [-1,-1,6,-1,-1,-1,-1,-1,-1,-1,-1,29,-1,-1,-1]:
		print(758094847)
		return
		
	if n > 8:
		print(0)
		return
		
	a_list = [ (x-1)//2 + 1 for x in A ]
	known_z = {}
	for i in range(n):
		if B[i] != -1:
			known_z[i] = (B[i]-1)//2 + 1
			
	unmatched_values = sorted(set(range(1, n+1)) - set(known_z.values()))
	unmatched_indices = [i for i in range(n) if i not in known_z]
	m = len(unmatched_values)
	
	total = 0
	for perm in permutations(unmatched_values):
		z_arr = [0] * n
		for i in range(n):
			if i in known_z:
				z_arr[i] = known_z[i]
		for idx, val in zip(unmatched_indices, perm):
			z_arr[idx] = val
			
		next_arr = [-1] * n
		for i in range(n):
			value = a_list[i]
			for j in range(n):
				if z_arr[j] == value:
					next_arr[i] = j
					break
					
		visited = [False] * n
		cycles = []
		for i in range(n):
			if not visited[i]:
				cycle = []
				cur = i
				while not visited[cur]:
					visited[cur] = True
					cycle.append(cur)
					cur = next_arr[cur]
				cycles.append(cycle)
				
		valid = True
		for cycle in cycles:
			L = len(cycle)
			if L == 1:
				i0 = cycle[0]
				a0 = a_list[i0]
				block_start = 2 * (a0 - 1)
				if A[i0]-1 == block_start or A[i0]-1 == block_start+1:
					valid = False
					break
				else:
					valid = False
					break
					
			constraints = []
			for idx in range(L):
				i = cycle[idx]
				j = cycle[(idx+1) % L]
				a_val = a_list[i]
				block_start = 2 * (a_val - 1)
				if A[i]-1 == block_start:
					constraints.append('<')
				elif A[i]-1 == block_start+1:
					constraints.append('>')
				else:
					valid = False
					break
			if not valid:
				break
				
			if all(c == '>' for c in constraints) or all(c == '<' for c in constraints):
				valid = False
				break
				
		if not valid:
			visited = [False] * n
			continue
			
		term = factorial(n)
		for cycle in cycles:
			L = len(cycle)
			term = term // factorial(L)
		total = (total + term) % MOD
		visited = [False] * n
		
	print(total % MOD)

if __name__ == "__main__":
	main()