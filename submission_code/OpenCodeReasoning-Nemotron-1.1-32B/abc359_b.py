def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	positions = [[] for _ in range(n+1)]
	
	for index, color in enumerate(arr):
		positions[color].append(index)
		
	count = 0
	for color in range(1, n+1):
		if len(positions[color]) == 2:
			if positions[color][1] - positions[color][0] == 2:
				count += 1
				
	print(count)

if __name__ == "__main__":
	main()