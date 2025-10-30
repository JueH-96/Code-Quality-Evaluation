def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	last_occurrence = {}
	B = []
	
	for i in range(n):
		if A[i] in last_occurrence:
			B.append(last_occurrence[A[i]] + 1)
		else:
			B.append(-1)
		last_occurrence[A[i]] = i
	
	print(" ".join(map(str, B)))

if __name__ == "__main__":
	main()