import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index]); index += 1
		K = int(data[index]); index += 1
		cakes = []
		for i in range(N):
			x = int(data[index]); y = int(data[index+1]); z = int(data[index+2]); index += 3
			cakes.append((x, y, z))
		
		s_list = [x + y + z for (x, y, z) in cakes]
		cakes_with_s = [(s_list[i], cakes[i][0], cakes[i][1], cakes[i][2]) for i in range(N)]
		cakes_with_s.sort(reverse=True)
		
		M = min(N, 2*K + 3)
		selected = cakes_with_s[:M]
		
		if M * (2*K) <= 10**7:
			dp = [[-10**18] * 8 for _ in range(2*K+1)]
			dp[0][0] = 0
			for i in range(M):
				s_i, X_i, Y_i, Z_i = selected[i]
				new_dp = [row[:] for row in dp]
				for j in range(0, 2*K):
					for mask in range(8):
						if dp[j][mask] == -10**18:
							continue
						nj = j + 1
						nm0 = mask ^ 1
						add0 = s_i - (Y_i + Z_i)
						if new_dp[nj][nm0] < dp[j][mask] + add0:
							new_dp[nj][nm0] = dp[j][mask] + add0
							
						nm1 = mask ^ 2
						add1 = s_i - (X_i + Z_i)
						if new_dp[nj][nm1] < dp[j][mask] + add1:
							new_dp[nj][nm1] = dp[j][mask] + add1
							
						nm2 = mask ^ 4
						add2 = s_i - (X_i + Y_i)
						if new_dp[nj][nm2] < dp[j][mask] + add2:
							new_dp[nj][nm2] = dp[j][mask] + add2
				dp = new_dp
			ans = dp[2*K][0]
			results.append(str(ans))
		else:
			selected_heuristic = cakes_with_s[:2*K]
			dp = [-10**18] * 8
			dp[0] = 0
			for i in range(2*K):
				s_i, X_i, Y_i, Z_i = selected_heuristic[i]
				new_dp = [-10**18] * 8
				for mask in range(8):
					if dp[mask] == -10**18:
						continue
					nm0 = mask ^ 1
					add0 = s_i - (Y_i + Z_i)
					if new_dp[nm0] < dp[mask] + add0:
						new_dp[nm0] = dp[mask] + add0
					nm1 = mask ^ 2
					add1 = s_i - (X_i + Z_i)
					if new_dp[nm1] < dp[mask] + add1:
						new_dp[nm1] = dp[mask] + add1
					nm2 = mask ^ 4
					add2 = s_i - (X_i + Y_i)
					if new_dp[nm2] < dp[mask] + add2:
						new_dp[nm2] = dp[mask] + add2
				dp = new_dp
			ans = dp[0]
			results.append(str(ans))
	print("
".join(results))

if __name__ == "__main__":
	main()