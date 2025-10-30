def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total_sum = sum(A)
	k = total_sum // n
	r = total_sum % n
	
	A_sorted = sorted(A)
	
	moves = 0
	for i in range(n - r):
		if A_sorted[i] > k:
			moves += A_sorted[i] - k
			
	for i in range(n - r, n):
		if A_sorted[i] > k + 1:
			moves += A_sorted[i] - (k + 1)
			
	print(moves)

if __name__ == "__main__":
	main()