import sys
from collections import Counter

def has_palindrome_seq(seq, k):
	n = len(seq)
	for i in range(0, n - k + 1):
		low = i
		high = i + k - 1
		valid = True
		while low < high:
			if seq[low] != seq[high]:
				valid = False
				break
			low += 1
			high -= 1
		if valid:
			return True
	return False

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n, k = map(int, data[0].split())
	s = data[1].strip()
	
	count_dict = Counter(s)
	total = [0]
	path = []
	
	def dfs():
		if len(path) == n:
			if not has_palindrome_seq(path, k):
				total[0] += 1
			return
		for c in list(count_dict.keys()):
			if count_dict[c] > 0:
				count_dict[c] -= 1
				path.append(c)
				dfs()
				path.pop()
				count_dict[c] += 1
				
	dfs()
	print(total[0])

if __name__ == "__main__":
	main()