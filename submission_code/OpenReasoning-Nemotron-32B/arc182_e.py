import math
import bisect
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	it = iter(data)
	N = int(next(it)); M = int(next(it)); C_val = int(next(it)); K = int(next(it))
	A = [int(next(it)) for _ in range(N)]
	
	if C_val == 0:
		base = min(A)
		print(base * K)
		return
		
	g = math.gcd(C_val, M)
	if g == 0:
		base = min(A)
		print(base * K)
		return
		
	m = M // g
	c = C_val // g
	
	if m > 10**6:
		group0 = []
		for a in A:
			if a % g == 0:
				a_prime = a // g
				group0.append(a_prime)
		group0_set = set(group0)
		if len(group0_set) == m:
			all_present = True
			for x in range(m):
				if x not in group0_set:
					all_present = False
					break
			if all_present:
				print(0)
				return
		print(0)
		return
	else:
		groups = [[] for _ in range(g)]
		for a in A:
			r = a % g
			a_prime = a // g
			groups[r].append(a_prime)
		for i in range(g):
			groups[i].sort()
		
		F_arr = [10**18] * m
		for t_val in range(m):
			for r in range(g):
				arr = groups[r]
				if not arr:
					continue
				idx = bisect.bisect_left(arr, m - t_val)
				candidate1 = 10**18
				candidate2 = 10**18
				if idx < len(arr):
					candidate1 = arr[idx] + t_val - m
				if idx > 0:
					candidate2 = arr[0] + t_val
				F_r_val = min(candidate1, candidate2)
				value = r + g * F_r_val
				if value < F_arr[t_val]:
					F_arr[t_val] = value
		
		full_cycles = K // m
		remainder = K % m
		total_sum = full_cycles * sum(F_arr)
		for k in range(remainder):
			t_val = (k * c) % m
			total_sum += F_arr[t_val]
		print(total_sum)

if __name__ == '__main__':
	main()