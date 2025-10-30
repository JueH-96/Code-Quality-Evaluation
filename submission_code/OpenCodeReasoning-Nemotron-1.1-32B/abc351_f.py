import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n < 2:
		print(0)
		return
		
	sorted_vals = sorted(set(A))
	comp = {val: idx+1 for idx, val in enumerate(sorted_vals)}
	m = len(sorted_vals)
	
	BIT_count = [0] * (m+1)
	BIT_sum = [0] * (m+1)
	
	def update(bit, index, delta):
		while index <= m:
			bit[index] += delta
			index += index & -index
			
	def query(bit, index):
		s = 0
		while index:
			s += bit[index]
			index -= index & -index
		return s
		
	total_ans = 0
	for i in range(n-1, -1, -1):
		x = A[i]
		pos = comp[x]
		
		total_count = query(BIT_count, m)
		total_sum = query(BIT_sum, m)
		
		prefix_count = query(BIT_count, pos)
		prefix_sum = query(BIT_sum, pos)
		
		count_greater = total_count - prefix_count
		sum_greater = total_sum - prefix_sum
		
		total_ans += (sum_greater - x * count_greater)
		
		update(BIT_count, pos, 1)
		update(BIT_sum, pos, x)
		
	print(total_ans)

if __name__ == "__main__":
	main()