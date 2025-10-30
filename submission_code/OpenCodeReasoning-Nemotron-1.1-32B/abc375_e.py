import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	n = int(data[0].strip())
	total = 0
	A = []
	B = []
	for i in range(1, 1 + n):
		parts = data[i].split()
		a = int(parts[0])
		b = int(parts[1])
		A.append(a)
		B.append(b)
		total += b

	if total % 3 != 0:
		print(-1)
		return

	target = total // 3
	sum1 = 0
	sum2 = 0
	for i in range(n):
		if A[i] == 1:
			sum1 += B[i]
		elif A[i] == 2:
			sum2 += B[i]
	d1 = sum1 - target
	d2 = sum2 - target

	dp = {}
	dp[(d1, d2)] = 0

	for i in range(n):
		new_dp = {}
		for state, moves in dp.items():
			d1_curr, d2_curr = state
			if (d1_curr, d2_curr) in new_dp:
				if new_dp[(d1_curr, d2_curr)] > moves:
					new_dp[(d1_curr, d2_curr)] = moves
			else:
				new_dp[(d1_curr, d2_curr)] = moves

			if A[i] == 1:
				new_d1 = d1_curr - B[i]
				new_d2 = d2_curr + B[i]
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

				new_d1 = d1_curr - B[i]
				new_d2 = d2_curr
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

			elif A[i] == 2:
				new_d1 = d1_curr + B[i]
				new_d2 = d2_curr - B[i]
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

				new_d1 = d1_curr
				new_d2 = d2_curr - B[i]
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

			else:
				new_d1 = d1_curr + B[i]
				new_d2 = d2_curr
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

				new_d1 = d1_curr
				new_d2 = d2_curr + B[i]
				new_state = (new_d1, new_d2)
				new_moves = moves + 1
				if new_state in new_dp:
					if new_dp[new_state] > new_moves:
						new_dp[new_state] = new_moves
				else:
					new_dp[new_state] = new_moves

		dp = new_dp

	if (0, 0) in dp:
		print(dp[(0, 0)])
	else:
		print(-1)

if __name__ == "__main__":
	main()