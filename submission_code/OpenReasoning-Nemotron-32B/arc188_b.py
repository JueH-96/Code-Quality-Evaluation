import math

T = int(input().strip())
results = []
for _ in range(T):
	N, K = map(int, input().split())
	
	if N % 2 == 0 and K == N // 2:
		if N == 2:
			results.append("Yes")
		else:
			results.append("No")
	else:
		g = math.gcd(2 * K, N)
		m = N // g
		fixed_points = [0, K]
		if N % 2 == 0:
			fixed_points.append(N // 2)
			fixed_points.append((K + N // 2) % N)
		
		residues = set()
		for x in fixed_points:
			residues.add(x % g)
		
		if len(residues) * m == N:
			results.append("Yes")
		else:
			results.append("No")

for res in results:
	print(res)