import sys
from collections import Counter

def main():
	data = list(map(int, sys.stdin.readline().split()))
	freq = Counter(data)
	freqs = sorted(freq.values(), reverse=True)
	
	if len(freqs) < 2:
		print("No")
	else:
		if freqs[0] >= 3 and freqs[1] >= 2:
			print("Yes")
		else:
			print("No")

if __name__ == "__main__":
	main()