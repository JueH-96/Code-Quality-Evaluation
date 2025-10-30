def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	i = 0
	j = n // 2
	count = 0
	
	while i < n // 2 and j < n:
		if 2 * A[i] <= A[j]:
			count += 1
			i += 1
			j += 1
		else:
			j += 1
			
	print(count)

if __name__ == "__main__":
	main()