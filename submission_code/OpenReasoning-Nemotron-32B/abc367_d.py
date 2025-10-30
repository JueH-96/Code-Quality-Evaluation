import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	M_val = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total_T = sum(A)
	r0 = (-total_T) % M_val
	
	P = [0] * n
	for i in range(1, n):
		P[i] = P[i-1] + A[i-1]
	
	arr = [(P[i], i) for i in range(n)]
	arr.sort(key=lambda x: (x[0], x[1]))
	
	freq = [0] * M_val
	for i in range(n):
		r = P[i] % M_val
		freq[r] += 1
		
	forward_arr = [0] * M_val
	backward_arr = freq[:]
	
	total_pairs = 0
	i = 0
	while i < n:
		j = i
		while j < n and arr[j][0] == arr[i][0]:
			j += 1
		group_size = j - i
		residue_group = arr[i][0] % M_val
		
		backward_arr[residue_group] -= group_size
		
		for k in range(i, j):
			pos_in_group = k - i
			count1 = forward_arr[residue_group] + pos_in_group
			target = (residue_group - r0 + M_val) % M_val
			count2 = backward_arr[target]
			total_pairs += count1 + count2
			forward_arr[residue_group] += 1
		
		i = j
		
	print(total_pairs)

if __name__ == '__main__':
	main()