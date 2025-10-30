import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	ops = []
	index = 2
	for i in range(m):
		L = int(data[index])
		R = int(data[index + 1])
		index += 2
		ops.append((L, R))
	
	found_cost1 = -1
	for i in range(m):
		L, R = ops[i]
		if L == 1 and R == n:
			found_cost1 = i
			break
			
	if found_cost1 != -1:
		print(1)
		res = []
		for j in range(m):
			if j == found_cost1:
				res.append('1')
			else:
				res.append('0')
		print(" ".join(res))
		return

	R1 = 0
	index1 = -1
	for i in range(m):
		L, R = ops[i]
		if L <= 1:
			if R > R1:
				R1 = R
				index1 = i
				
	candidate_j = -1
	if R1 > 0:
		for i in range(m):
			L, R = ops[i]
			if L <= R1 + 1 and R >= n:
				candidate_j = i
				break
				
	if candidate_j != -1:
		print(2)
		res = []
		for j in range(m):
			if j == index1 or j == candidate_j:
				res.append('1')
			else:
				res.append('0')
		print(" ".join(res))
		return
		
	buckets = [[] for _ in range(n + 1)]
	for i in range(m):
		L, R = ops[i]
		if L <= n:
			buckets[L].append((R, i))
			
	best_val_arr = [0] * (n + 1)
	best_index_arr = [-1] * (n + 1)
	second_val_arr = [0] * (n + 1)
	second_index_arr = [-1] * (n + 1)
	
	best_val0 = 0
	best_index0 = -1
	second_val0 = 0
	second_index0 = -1
	
	for L in range(1, n + 1):
		bv = best_val0
		bi = best_index0
		sv = second_val0
		si = second_index0
		
		for (R, idx) in buckets[L]:
			if R > bv:
				sv = bv
				si = bi
				bv = R
				bi = idx
			elif R == bv:
				if sv < bv:
					sv = bv
					si = idx
			elif R > sv:
				sv = R
				si = idx
				
		best_val0 = bv
		best_index0 = bi
		second_val0 = sv
		second_index0 = si
		
		best_val_arr[L] = bv
		best_index_arr[L] = bi
		second_val_arr[L] = sv
		second_index_arr[L] = si

	found_i = -1
	found_j = -1
	for i in range(m):
		L, R = ops[i]
		if L > n:
			continue
		bv = best_val_arr[L]
		bi = best_index_arr[L]
		sv = second_val_arr[L]
		si = second_index_arr[L]
		
		if R < bv:
			found_i = i
			found_j = bi
			break
		elif R == bv:
			if sv == bv:
				if bi != i:
					found_i = i
					found_j = bi
					break
				else:
					found_i = i
					found_j = si
					break
			else:
				if bi != i:
					found_i = i
					found_j = bi
					break
					
	if found_i != -1:
		print(2)
		res = []
		for j in range(m):
			if j == found_i:
				res.append('2')
			elif j == found_j:
				res.append('1')
			else:
				res.append('0')
		print(" ".join(res))
		return
		
	print(-1)

if __name__ == '__main__':
	main()