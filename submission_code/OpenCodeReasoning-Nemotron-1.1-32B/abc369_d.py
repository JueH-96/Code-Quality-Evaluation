def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	if n == 0:
		print(0)
		return
		
	dp0 = 0
	dp1 = arr[0]
	
	for i in range(1, n):
		next_dp0 = max(dp0, dp1 + 2 * arr[i])
		next_dp1 = max(dp1, dp0 + arr[i])
		dp0, dp1 = next_dp0, next_dp1
		
	print(max(dp0, dp1))

if __name__ == "__main__":
	main()