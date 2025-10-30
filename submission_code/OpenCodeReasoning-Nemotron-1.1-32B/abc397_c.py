import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	prefix = [0] * (n + 1)
	freq = {}
	distinct = 0
	for i in range(1, n + 1):
		x = A[i - 1]
		if x not in freq:
			freq[x] = 0
			distinct += 1
		freq[x] += 1
		prefix[i] = distinct
		
	suffix = [0] * (n + 1)
	freq = {}
	distinct = 0
	for i in range(n - 1, -1, -1):
		x = A[i]
		if x not in freq:
			freq[x] = 0
			distinct += 1
		freq[x] += 1
		suffix[i] = distinct
		
	ans = 0
	for i in range(1, n):
		total = prefix[i] + suffix[i]
		if total > ans:
			ans = total
			
	print(ans)

if __name__ == '__main__':
	main()