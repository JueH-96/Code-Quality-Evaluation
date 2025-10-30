def main():
	n = int(input().strip())
	scores = list(map(int, input().split()))
	
	arr = [(scores[i], i) for i in range(n)]
	arr.sort(key=lambda x: x[0], reverse=True)
	
	res = [0] * n
	current_rank = 1
	i = 0
	while i < n:
		j = i
		current_score = arr[i][0]
		while j < n and arr[j][0] == current_score:
			j += 1
		count = j - i
		for k in range(i, j):
			original_index = arr[k][1]
			res[original_index] = current_rank
		current_rank += count
		i = j
		
	for rank in res:
		print(rank)

if __name__ == '__main__':
	main()