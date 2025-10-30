n = int(input())
A = list(map(int, input().split()))
ops = []
for _ in range(n-1):
	s, t = map(int, input().split())
	ops.append((s, t))

flow = 0
for i in range(n-1):
	available = A[i] + flow
	s, t = ops[i]
	x = available // s
	flow = t * x

print(A[-1] + flow)