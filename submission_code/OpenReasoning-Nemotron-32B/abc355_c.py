def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	t = int(data[1])
	a_list = list(map(int, data[2:2+t]))
	
	row_count = [0] * n
	col_count = [0] * n
	diag1_count = 0
	diag2_count = 0
	
	for i in range(t):
		x = a_list[i]
		r = (x - 1) // n
		c = (x - 1) % n
		
		row_count[r] += 1
		col_count[c] += 1
		
		if r == c:
			diag1_count += 1
		if r + c == n - 1:
			diag2_count += 1
			
		if row_count[r] == n or col_count[c] == n or diag1_count == n or diag2_count == n:
			print(i + 1)
			return
			
	print(-1)

if __name__ == "__main__":
	main()