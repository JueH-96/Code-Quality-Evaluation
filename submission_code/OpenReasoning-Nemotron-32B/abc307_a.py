def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	weeks = []
	for i in range(n):
		start_index = i * 7
		end_index = start_index + 7
		week_data = arr[start_index:end_index]
		weeks.append(week_data)
	
	sums = [sum(week) for week in weeks]
	
	print(" ".join(map(str, sums)))

if __name__ == "__main__":
	main()