def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	
	n = int(data[0].strip())
	people = []
	total_sum = 0
	for i in range(1, n+1):
		parts = data[i].split()
		if len(parts) < 2:
			continue
		a = int(parts[0])
		b = int(parts[1])
		people.append((a, b))
		total_sum += b

	if total_sum % 3 != 0:
		print(-1)
		return
		
	T = total_sum // 3

	INF = 10**9
	dp = [[INF] * (T+1) for _ in range(T+1)]
	dp[0][0] = 0
	current_total = 0

	for a, b in people:
		current_total_prev = current_total
		current_total += b
		new_dp = [[INF] * (T+1) for _ in range(T+1)]
		
		s1_max = min(T, current_total_prev)
		for s1 in range(0, s1_max + 1):
			s2_max = min(T, current_total_prev - s1)
			for s2 in range(0, s2_max + 1):
				if dp[s1][s2] == INF:
					continue
				s3_prev = current_total_prev - s1 - s2
				if s3_prev > T:
					continue
				
				cost_stay = 0 if a == 1 else 1
				if s1 + b <= T:
					new_cost = dp[s1][s2] + cost_stay
					if new_cost < new_dp[s1+b][s2]:
						new_dp[s1+b][s2] = new_cost
				
				cost_stay = 0 if a == 2 else 1
				if s2 + b <= T:
					new_cost = dp[s1][s2] + cost_stay
					if new_cost < new_dp[s1][s2+b]:
						new_dp[s1][s2+b] = new_cost
				
				cost_stay = 0 if a == 3 else 1
				if s3_prev + b <= T:
					new_cost = dp[s1][s2] + cost_stay
					if new_cost < new_dp[s1][s2]:
						new_dp[s1][s2] = new_cost
						
		dp = new_dp

	if dp[T][T] == INF:
		print(-1)
	else:
		print(dp[T][T])

if __name__ == "__main__":
	main()