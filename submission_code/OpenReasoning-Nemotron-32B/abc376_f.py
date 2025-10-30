import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	it = iter(data)
	N = int(next(it))
	Q = int(next(it))
	instructions = []
	for _ in range(Q):
		h = next(it)
		t = int(next(it))
		instructions.append((h, t-1))
	
	INF = 10**18
	dp = [[INF] * N for _ in range(N)]
	dp[0][1] = 0
	
	for h, t in instructions:
		dist = [[INF] * N for _ in range(N)]
		q = deque()
		for a in range(N):
			for b in range(N):
				if dp[a][b] != INF:
					dist[a][b] = dp[a][b]
					q.append((a, b))
		
		while q:
			a, b = q.popleft()
			current_cost = dist[a][b]
			moves = [
				((a + 1) % N, b),
				((a - 1 + N) % N, b),
				(a, (b + 1) % N),
				(a, (b - 1 + N) % N)
			]
			for na, nb in moves:
				if na == nb:
					continue
				if na == b or nb == a:
					continue
				if dist[na][nb] > current_cost + 1:
					dist[na][nb] = current_cost + 1
					q.append((na, nb))
		
		new_dp = [[INF] * N for _ in range(N)]
		if h == 'L':
			for b in range(N):
				if b != t:
					new_dp[t][b] = dist[t][b]
		else:
			for a in range(N):
				if a != t:
					new_dp[a][t] = dist[a][t]
		dp = new_dp
	
	ans = INF
	for a in range(N):
		for b in range(N):
			if dp[a][b] < ans:
				ans = dp[a][b]
	print(ans)

if __name__ == '__main__':
	main()