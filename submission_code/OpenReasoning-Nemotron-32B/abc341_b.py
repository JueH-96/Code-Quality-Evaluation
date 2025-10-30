n = int(input().strip())
A = list(map(int, input().split()))
edges = []
for _ in range(n-1):
	s, t = map(int, input().split())
	edges.append((s, t))

x = A[0]
for i in range(n-1):
	s, t = edges[i]
	k = x // s
	x = A[i+1] + k * t

print(x)