import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 0:
		print(0)
		return
		
	max_len = 0
	
	l = 0
	freq = {}
	for r in range(0, n, 2):
		if r + 1 >= n:
			break
		if A[r] != A[r+1]:
			current_length = r - l
			if current_length > max_len:
				max_len = current_length
			l = r + 2
			freq = {}
		else:
			while freq.get(A[r], 0) > 0:
				freq[A[l]] -= 1
				l += 2
			freq[A[r]] = freq.get(A[r], 0) + 1
			current_length = r + 1 - l + 1
			if current_length > max_len:
				max_len = current_length
				
	l = 1
	freq = {}
	for r in range(1, n, 2):
		if r + 1 >= n:
			break
		if A[r] != A[r+1]:
			current_length = r - l
			if current_length > max_len:
				max_len = current_length
			l = r + 2
			freq = {}
		else:
			while freq.get(A[r], 0) > 0:
				freq[A[l]] -= 1
				l += 2
			freq[A[r]] = freq.get(A[r], 0) + 1
			current_length = r + 1 - l + 1
			if current_length > max_len:
				max_len = current_length
				
	print(max_len)

if __name__ == "__main__":
	main()