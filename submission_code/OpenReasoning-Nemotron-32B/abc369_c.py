def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	total = n
	
	if n > 1:
		diff = []
		for i in range(1, n):
			diff.append(A[i] - A[i-1])
		
		i = 0
		n_diff = len(diff)
		while i < n_diff:
			j = i
			current_diff = diff[i]
			while j < n_diff and diff[j] == current_diff:
				j += 1
			L = j - i
			total += L * (L + 1) // 2
			i = j
			
	print(total)

if __name__ == '__main__':
	main()