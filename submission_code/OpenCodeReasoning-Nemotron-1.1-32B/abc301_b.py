def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	while True:
		n = len(A)
		found_index = -1
		for i in range(n - 1):
			if abs(A[i] - A[i + 1]) != 1:
				found_index = i
				break
				
		if found_index == -1:
			break
			
		if A[found_index] < A[found_index + 1]:
			insert_list = list(range(A[found_index] + 1, A[found_index + 1]))
		else:
			insert_list = list(range(A[found_index] - 1, A[found_index + 1], -1))
			
		A = A[:found_index + 1] + insert_list + A[found_index + 1:]
		
	print(" ".join(map(str, A)))

if __name__ == "__main__":
	main()