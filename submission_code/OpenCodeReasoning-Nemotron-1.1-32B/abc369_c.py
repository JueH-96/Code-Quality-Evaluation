def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	if n == 0:
		print(0)
		return
		
	total = 1
	prev = 1
	
	for i in range(1, n):
		if i >= 2 and arr[i] - arr[i-1] == arr[i-1] - arr[i-2]:
			current = prev + 1
		else:
			current = 2
		total += current
		prev = current
		
	print(total)

if __name__ == "__main__":
	main()