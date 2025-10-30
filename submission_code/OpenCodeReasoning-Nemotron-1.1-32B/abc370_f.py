import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total = sum(A)
	low = min(A)
	high = total // k
	
	B = A + A
	n2 = 2 * n
	P = [0] * (n2 + 1)
	for i in range(1, n2 + 1):
		P[i] = P[i - 1] + B[i - 1]
	
	LOG = (n).bit_length()
	ans_x = low
	while low <= high:
		mid = (low + high) // 2
		f_arr = [0] * n2
		j = 0
		cur_sum = 0
		for i in range(n2):
			while j < n2 and cur_sum < mid:
				cur_sum += B[j]
				j += 1
			if cur_sum >= mid:
				f_arr[i] = j - 1
			else:
				f_arr[i] = n2
			cur_sum -= B[i]
		
		g = [0] * n2
		for i in range(n2):
			if f_arr[i] < n2:
				g[i] = f_arr[i] + 1
			else:
				g[i] = n2
		
		dp_table = [[0] * n2 for _ in range(LOG)]
		for i in range(n2):
			dp_table[0][i] = g[i]
		
		for i in range(1, LOG):
			for j in range(n2):
				if dp_table[i - 1][j] < n2:
					dp_table[i][j] = dp_table[i - 1][dp_table[i - 1][j]]
				else:
					dp_table[i][j] = n2
		
		found = False
		for start in range(n):
			current = start
			steps = k - 1
			for j in range(LOG):
				if steps & (1 << j):
					if current < n2:
						current = dp_table[j][current]
					else:
						break
			if current < start + n:
				last_seg_sum = P[start + n] - P[current]
			else:
				last_seg_sum = 0
			if last_seg_sum >= mid:
				found = True
				break
		
		if found:
			ans_x = mid
			low = mid + 1
		else:
			high = mid - 1
	
	f_arr = [0] * n2
	j = 0
	cur_sum = 0
	for i in range(n2):
		while j < n2 and cur_sum < ans_x:
			cur_sum += B[j]
			j += 1
		if cur_sum >= ans_x:
			f_arr[i] = j - 1
		else:
			f_arr[i] = n2
		cur_sum -= B[i]
	
	g = [0] * n2
	for i in range(n2):
		if f_arr[i] < n2:
			g[i] = f_arr[i] + 1
		else:
			g[i] = n2
	
	dp_table = [[0] * n2 for _ in range(LOG)]
	for i in range(n2):
		dp_table[0][i] = g[i]
	
	for i in range(1, LOG):
		for j in range(n2):
			if dp_table[i - 1][j] < n2:
				dp_table[i][j] = dp_table[i - 1][dp_table[i - 1][j]]
			else:
				dp_table[i][j] = n2
	
	y = 0
	for i in range(n):
		current = i + 1
		steps = k - 1
		for j in range(LOG):
			if steps & (1 << j):
				if current < n2:
					current = dp_table[j][current]
				else:
					break
		if current < i + n + 1:
			last_seg_sum = P[i + n + 1] - P[current]
		else:
			last_seg_sum = 0
		if last_seg_sum < ans_x:
			y += 1
	
	print(f"{ans_x} {y}")

if __name__ == "__main__":
	main()