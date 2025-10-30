def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	first_occurrence = [0] * (n + 1)
	count = 0
	
	for idx, color in enumerate(arr):
		if first_occurrence[color] == 0:
			first_occurrence[color] = idx + 1
		else:
			gap = (idx + 1) - first_occurrence[color] - 1
			if gap == 1:
				count += 1
				
	print(count)

if __name__ == '__main__':
	main()