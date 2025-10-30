import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	courses = []
	index = 1
	for i in range(n):
		x = int(data[index])
		y = int(data[index + 1])
		index += 2
		courses.append((x, y))
	
	NEG_INF = -10**18
	dp0 = 0
	dp1 = NEG_INF
	
	for x, y in courses:
		if x == 0:
			next_dp0 = max(dp0, dp0 + y, dp1 + y)
			next_dp1 = dp1
		else:
			next_dp0 = dp0
			next_dp1 = max(dp1, dp0 + y)
		
		dp0, dp1 = next_dp0, next_dp1
		
	ans = max(dp0, dp1)
	print(ans)

if __name__ == '__main__':
	main()