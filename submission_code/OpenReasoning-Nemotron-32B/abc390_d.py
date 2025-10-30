import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	s = set()
	arr = []

	def dfs(i):
		if i == n:
			xor_val = 0
			for x in arr:
				xor_val ^= x
			s.add(xor_val)
			return
		
		arr.append(A[i])
		dfs(i + 1)
		arr.pop()
		
		for j in range(len(arr)):
			arr[j] += A[i]
			dfs(i + 1)
			arr[j] -= A[i]
			
	dfs(0)
	print(len(s))

if __name__ == '__main__':
	main()