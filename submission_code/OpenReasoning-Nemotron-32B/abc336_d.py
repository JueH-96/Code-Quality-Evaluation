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
		
	L = [0] * n
	R = [0] * n
	
	L[0] = 1
	for i in range(1, n):
		L[i] = min(A[i], L[i-1] + 1)
		
	R[n-1] = 1
	for i in range(n-2, -1, -1):
		R[i] = min(A[i], R[i+1] + 1)
		
	ans = 0
	for i in range(n):
		candidate = min(L[i], R[i])
		if candidate > ans:
			ans = candidate
			
	print(ans)

if __name__ == "__main__":
	main()