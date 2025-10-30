import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	Q = int(next(it))
	
	INF = 10**18
	dp = [[INF] * N for _ in range(N)]
	dp[0][1] = 0
	
	for _ in range(Q):
		H = next(it)
		T = int(next(it)) - 1
		
		dist = [[INF] * N for _ in range(N)]
		heap = []
		for l in range(N):
			for r in range(N):
				if dp[l][r] < INF:
					dist[l][r] = dp[l][r]
					heapq.heappush(heap, (dp[l][r], l, r))
		
		while heap:
			cost, l, r = heapq.heappop(heap)
			if cost != dist[l][r]:
				continue
			for dl in [1, -1]:
				nl = (l + dl) % N
				if nl == r:
					continue
				new_cost = cost + 1
				if new_cost < dist[nl][r]:
					dist[nl][r] = new_cost
					heapq.heappush(heap, (new_cost, nl, r))
			for dr in [1, -1]:
				nr = (r + dr) % N
				if nr == l:
					continue
				new_cost = cost + 1
				if new_cost < dist[l][nr]:
					dist[l][nr] = new_cost
					heapq.heappush(heap, (new_cost, l, nr))
		
		new_dp = [[INF] * N for _ in range(N)]
		for l in range(N):
			for r in range(N):
				if (H == 'L' and l == T) or (H == 'R' and r == T):
					new_dp[l][r] = dist[l][r]
		dp = new_dp

	ans = INF
	for l in range(N):
		for r in range(N):
			if dp[l][r] < ans:
				ans = dp[l][r]
	print(ans)

if __name__ == '__main__':
	main()