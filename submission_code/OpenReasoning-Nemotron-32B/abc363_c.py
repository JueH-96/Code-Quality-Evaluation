import sys
from collections import Counter

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	parts = data[0].split()
	if not parts:
		print(0)
		return
	N = int(parts[0])
	K = int(parts[1])
	S = data[1].strip()
	
	if K > N:
		print(0)
		return
		
	freq = Counter(S)
	
	def is_palindrome(sub):
		return sub == sub[::-1]
	
	def dfs(curr, freq):
		if len(curr) == N:
			return 1
			
		total = 0
		available_chars = [char for char in freq if freq[char] > 0]
		for char in available_chars:
			curr.append(char)
			freq[char] -= 1
			
			if len(curr) >= K:
				if is_palindrome(curr[-K:]):
					curr.pop()
					freq[char] += 1
					continue
					
			total += dfs(curr, freq)
			
			curr.pop()
			freq[char] += 1
			
		return total
		
	result = dfs([], freq)
	print(result)

if __name__ == "__main__":
	main()