import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+n+n-1]))
	
	A.sort()
	B.sort()
	
	prefix = [True] * (n+1)
	for i in range(1, n):
		prefix[i] = prefix[i-1] and (A[i-1] <= B[i-1])
	
	suffix = [True] * (n+1)
	for i in range(n-2, -1, -1):
		suffix[i] = (A[i+1] <= B[i]) and suffix[i+1]
	
	ans = 10**18
	found = False
	for i in range(n):
		if prefix[i] and suffix[i]:
			if A[i] < ans:
				ans = A[i]
			found = True
	
	print(ans if found else -1)

if __name__ == "__main__":
	main()