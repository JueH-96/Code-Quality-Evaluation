def main():
	data = input().split()
	nums = list(map(int, data))
	freq = [0] * 5
	for num in nums:
		freq[num] += 1
	total_operations = 0
	for i in range(1, 5):
		total_operations += freq[i] // 2
	print(total_operations)

if __name__ == "__main__":
	main()