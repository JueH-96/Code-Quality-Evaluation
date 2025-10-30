import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		n = int(data[index]); index += 1
		A = list(map(int, data[index:index+n]))
		index += n
		s = sum(A)
		if n == 1:
			results.append("Yes")
			continue
			
		prefix = [0] * n
		prefix[0] = A[0]
		for i in range(1, n):
			prefix[i] = prefix[i-1] + A[i]
			
		low = 0
		high = s - A[0]
		found = False
		
		while low <= high:
			mid = (low + high) // 2
			f0 = 0
			f1 = A[0] + mid
			for i in range(2, n):
				F_i = max(prefix[i-1], 2 * f1 - f0)
				f0, f1 = f1, F_i
				
			if s >= 2 * f1 - f0:
				found = True
				break
			else:
				low = mid + 1
				
		results.append("Yes" if found else "No")
		
	for res in results:
		print(res)

if __name__ == "__main__":
	main()