def main():
	S = input().strip()
	freq = {}
	for char in S:
		freq[char] = freq.get(char, 0) + 1
	max_count = max(freq.values())
	answer = min(char for char in freq if freq[char] == max_count)
	print(answer)

if __name__ == "__main__":
	main()