def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	res = [-1] * n
	last_occurrence = {}
	
	for i in range(n):
		num = arr[i]
		if num in last_occurrence:
			res[i] = last_occurrence[num] + 1
		else:
			res[i] = -1
		last_occurrence[num] = i
	
	print(" ".join(map(str, res)))

if __name__ == '__main__':
	main()