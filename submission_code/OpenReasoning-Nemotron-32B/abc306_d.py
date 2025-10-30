import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	courses = []
	for i in range(1, n+1):
		parts = data[i].split()
		x = int(parts[0])
		y = int(parts[1])
		courses.append((x, y))
	
	dp0 = 0
	dp1 = -10**18
	
	for i in range(n):
		x, y = courses[i]
		next_dp0 = -10**18
		next_dp1 = -10**18
		
		next_dp0 = max(next_dp0, dp0)
		next_dp1 = max(next_dp1, dp1)
		
		if x == 0:
			candidate0 = dp0 + y
			candidate1 = dp1 + y
			next_dp0 = max(next_dp0, candidate0, candidate1)
		else:
			candidate0 = dp0 + y
			next_dp1 = max(next_dp1, candidate0)
		
		dp0 = next_dp0
		dp1 = next_dp1
	
	ans = max(dp0, dp1)
	print(ans)

if __name__ == "__main__":
	main()