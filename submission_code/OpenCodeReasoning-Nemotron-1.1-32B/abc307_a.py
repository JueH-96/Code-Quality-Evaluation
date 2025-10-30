def main():
	n = int(input().strip())
	data = list(map(int, input().split()))
	
	results = []
	for i in range(n):
		start_index = i * 7
		week_data = data[start_index:start_index + 7]
		total = sum(week_data)
		results.append(str(total))
	
	print(" ".join(results))

if __name__ == "__main__":
	main()