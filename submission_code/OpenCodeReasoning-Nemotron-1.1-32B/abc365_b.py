def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	max1 = -10**18
	max2 = -10**18
	idx1 = -1
	idx2 = -1
	
	for i in range(n):
		if arr[i] > max1:
			max2 = max1
			idx2 = idx1
			max1 = arr[i]
			idx1 = i
		elif arr[i] > max2:
			max2 = arr[i]
			idx2 = i
			
	print(idx2 + 1)

if __name__ == "__main__":
	main()