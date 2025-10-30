import sys
from collections import Counter

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	freq = Counter(A)
	
	candidate = None
	for idx, num in enumerate(A):
		if freq[num] == 1:
			if candidate is None or num > candidate[0]:
				candidate = (num, idx + 1)
				
	if candidate is None:
		print(-1)
	else:
		print(candidate[1])

if __name__ == "__main__":
	main()