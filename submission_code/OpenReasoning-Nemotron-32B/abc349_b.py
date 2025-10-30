from collections import Counter

def main():
	s = input().strip()
	char_freq = Counter(s)
	freq_of_freq = Counter(char_freq.values())
	for count_val in freq_of_freq.values():
		if count_val != 2:
			print("No")
			return
	print("Yes")

if __name__ == "__main__":
	main()