import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 0:
		print(0)
		return
		
	L = [0] * n
	L[0] = 1
	for i in range(1, n):
		L[i] = min(L[i-1] + 1, A[i])
	
	R = [0] * n
	R[-1] = 1
	for i in range(n-2, -1, -1):
		R[i] = min(R[i+1] + 1, A[i])
	
	ans = 0
	for i in range(n):
		k_i = min(L[i], R[i])
		if k_i > ans:
			ans = k_i
			
	print(ans)

if __name__ == '__main__':
	main()