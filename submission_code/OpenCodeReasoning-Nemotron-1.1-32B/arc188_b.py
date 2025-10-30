import math

T = int(input().strip())
for _ in range(T):
	N, K = map(int, input().split())
	g = math.gcd(2 * K, N)
	if g == 1 or g == 2:
		print("Yes")
	else:
		print("No")