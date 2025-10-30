def main():
	S = input().strip()
	freq = {}
	for char in S:
		freq[char] = freq.get(char, 0) + 1
		
	max_freq = max(freq.values())
	candidates = [char for char in freq if freq[char] == max_freq]
	answer = min(candidates)
	print(answer)

if __name__ == "__main__":
	main()