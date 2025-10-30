def main():
	data = input().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, input().split()))
	
	starts = 0
	current_empty = k
	i = 0
	while i < n:
		if current_empty < A[i]:
			starts += 1
			current_empty = k
		else:
			current_empty -= A[i]
			i += 1
			
	starts += 1
	print(starts)

if __name__ == "__main__":
	main()