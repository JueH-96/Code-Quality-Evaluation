import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	H = int(data[0])
	W = int(data[1])
	N = int(data[2])
	holed = [[False] * W for _ in range(H)]
	
	index = 3
	for _ in range(N):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		holed[a-1][b-1] = True
		
	dp0 = [0] * W
	dp1 = [0] * W
	total = 0
	
	for i in range(H-1, -1, -1):
		holed_row = holed[i]
		dp0_local = dp0
		dp1_local = dp1
		diag = 0
		next_right = 0
		row_sum = 0
		for j in range(W-1, -1, -1):
			current_dp0_j = dp0_local[j]
			if holed_row[j]:
				val = 0
				next_right = 0
				diag = current_dp0_j
			else:
				if i == H-1 or j == W-1:
					val = 1
					next_right = 1
					diag = current_dp0_j
				else:
					val = min(current_dp0_j, next_right, diag) + 1
					next_right = val
					diag = current_dp0_j
			dp1_local[j] = val
			row_sum += val
		total += row_sum
		dp0, dp1 = dp1, dp0
		
	print(total)

if __name__ == "__main__":
	main()