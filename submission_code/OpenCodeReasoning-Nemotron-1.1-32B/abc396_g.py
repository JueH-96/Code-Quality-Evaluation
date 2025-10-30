import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W = map(int, data[0].split())
	grid = data[1:1+H]
	
	T = 0
	M = []
	for i in range(H):
		s = grid[i].strip()
		row = []
		for char in s:
			if char == '1':
				T += 1
				row.append(-1)
			else:
				row.append(1)
		M.append(row)
	
	a = [sum(row) for row in M]
	
	total_masks = 1 << W
	masks_list = []
	for mask in range(total_masks):
		cnt = bin(mask).count('1')
		if cnt > W // 2:
			continue
		bits = []
		for j in range(W):
			if mask & (1 << j):
				bits.append(j)
		masks_list.append((mask, bits))
	
	best = float('inf')
	for mask, bits in masks_list:
		total_variable = 0
		for i in range(H):
			x = 0
			for j in bits:
				x += M[i][j]
			total_variable += min(x, a[i] - x)
		total_sum_candidate = T + total_variable
		if total_sum_candidate < best:
			best = total_sum_candidate
			
	print(best)

if __name__ == '__main__':
	main()