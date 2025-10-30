from collections import Counter

def main():
	s = input().strip()
	freq = Counter(s)
	count_freq = Counter(freq.values())
	max_freq = max(freq.values()) if freq else 0
	
	for i in range(1, max_freq + 1):
		if count_freq.get(i, 0) not in [0, 2]:
			print("No")
			return
	
	print("Yes")

if __name__ == '__main__':
	main()