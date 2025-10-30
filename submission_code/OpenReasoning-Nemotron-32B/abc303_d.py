import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	parts = data[0].split()
	if not parts:
		print(0)
		return
	X = int(parts[0])
	Y = int(parts[1])
	Z = int(parts[2])
	S = data[1].strip() if len(data) > 1 else ""
	n = len(S)
	if n == 0:
		print(0)
		return

	if S[0] == 'a':
		dp0 = X
		dp1 = Z + Y
	else:
		dp0 = Y
		dp1 = Z + X

	for i in range(1, n):
		c = S[i]
		if c == 'a':
			cost0 = X
			cost1 = Y
		else:
			cost0 = Y
			cost1 = X
		
		new_dp0 = min(dp0 + cost0, dp1 + Z + cost0)
		new_dp1 = min(dp1 + cost1, dp0 + Z + cost1)
		
		dp0, dp1 = new_dp0, new_dp1

	print(min(dp0, dp1))

if __name__ == "__main__":
	main()