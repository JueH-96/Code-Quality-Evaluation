import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	X = int(data[1])
	lists = {1: [], 2: [], 3: []}
	index = 2
	for i in range(n):
		v = int(data[index])
		a = int(data[index+1])
		c = int(data[index+2])
		index += 3
		lists[v].append((a, c))
	
	dp1 = [0] * (X+1)
	dp2 = [0] * (X+1)
	dp3 = [0] * (X+1)
	
	for a, c in lists[1]:
		for cal in range(X, c-1, -1):
			if dp1[cal] < dp1[cal-c] + a:
				dp1[cal] = dp1[cal-c] + a
				
	for a, c in lists[2]:
		for cal in range(X, c-1, -1):
			if dp2[cal] < dp2[cal-c] + a:
				dp2[cal] = dp2[cal-c] + a
				
	for a, c in lists[3]:
		for cal in range(X, c-1, -1):
			if dp3[cal] < dp3[cal-c] + a:
				dp3[cal] = dp3[cal-c] + a
				
	M1 = dp1[X]
	M2 = dp2[X]
	M3 = dp3[X]
	M = min(M1, M2, M3)
	
	def min_cal(dp, a):
		if a == 0:
			return 0
		for cal in range(0, X+1):
			if dp[cal] >= a:
				return cal
		return X + 1
	
	low, high = 0, M
	ans = 0
	while low <= high:
		mid = (low + high) // 2
		c1 = min_cal(dp1, mid)
		c2 = min_cal(dp2, mid)
		c3 = min_cal(dp3, mid)
		total_cal = c1 + c2 + c3
		if total_cal <= X:
			ans = mid
			low = mid + 1
		else:
			high = mid - 1
			
	print(ans)

if __name__ == "__main__":
	main()