import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	max_val = 200000

	min_index_for_value = [n] * (max_val + 1)
	
	for i in range(n):
		a = A[i]
		if a <= max_val:
			if i < min_index_for_value[a]:
				min_index_for_value[a] = i
				
	res_arr = [0] * (max_val + 1)
	current_min = n
	for x in range(0, max_val + 1):
		if min_index_for_value[x] < current_min:
			current_min = min_index_for_value[x]
		res_arr[x] = current_min
		
	output_lines = []
	for b in B:
		if b < 0:
			b = 0
		if b > max_val:
			b = max_val
		idx = res_arr[b]
		if idx == n:
			output_lines.append("-1")
		else:
			output_lines.append(str(idx + 1))
			
	print("
".join(output_lines))

if __name__ == "__main__":
	main()