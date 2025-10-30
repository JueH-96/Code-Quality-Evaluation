import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	X, Y, Z = map(int, data[0].split())
	S = data[1].strip()
	
	dp0 = 0
	dp1 = 10**18
	
	for c in S:
		if c == 'a':
			new_dp0 = min(dp0 + X, dp1 + Z + X)
			new_dp1 = min(dp1 + Y, dp0 + Z + Y)
		else:
			new_dp0 = min(dp0 + Y, dp1 + Z + Y)
			new_dp1 = min(dp1 + X, dp0 + Z + X)
		dp0, dp1 = new_dp0, new_dp1
		
	print(min(dp0, dp1))

if __name__ == "__main__":
	main()