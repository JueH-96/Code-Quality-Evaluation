import sys
from collections import Counter

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	freq = Counter(arr)
	
	max_val = -1
	max_index = -1
	for i in range(n):
		if freq[arr[i]] == 1:
			if arr[i] > max_val:
				max_val = arr[i]
				max_index = i + 1
				
	print(max_index if max_index != -1 else -1)

if __name__ == '__main__':
	main()