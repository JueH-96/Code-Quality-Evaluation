import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	it = iter(data)
	n = int(next(it))
	X = int(next(it))
	groups = {1: [], 2: [], 3: []}
	for _ in range(n):
		v = int(next(it))
		a = int(next(it))
		c = int(next(it))
		groups[v].append((a, c))
	
	total1 = sum(a for a, c in groups[1])
	total2 = sum(a for a, c in groups[2])
	total3 = sum(a for a, c in groups[3])
	M = min(total1, total2, total3)
	
	dp1 = [0] * (X + 1)
	for a, c in groups[1]:
		for j in range(X, c - 1, -1):
			if dp1[j - c] + a > dp1[j]:
				dp1[j] = dp1[j - c] + a
	for j in range(1, X + 1):
		dp1[j] = max(dp1[j], dp1[j - 1])
	
	dp2 = [0] * (X + 1)
	for a, c in groups[2]:
		for j in range(X, c - 1, -1):
			if dp2[j - c] + a > dp2[j]:
				dp2[j] = dp2[j - c] + a
	for j in range(1, X + 1):
		dp2[j] = max(dp2[j], dp2[j - 1])
	
	dp3 = [0] * (X + 1)
	for a, c in groups[3]:
		for j in range(X, c - 1, -1):
			if dp3[j - c] + a > dp3[j]:
				dp3[j] = dp3[j - c] + a
	for j in range(1, X + 1):
		dp3[j] = max(dp3[j], dp3[j - 1])
	
	low, high = 0, M
	ans = 0
	while low <= high:
		mid = (low + high) // 2
		H1 = 10**18
		for cal in range(X + 1):
			if dp1[cal] >= mid:
				H1 = cal
				break
		H2 = 10**18
		for cal in range(X + 1):
			if dp2[cal] >= mid:
				H2 = cal
				break
		H3 = 10**18
		for cal in range(X + 1):
			if dp3[cal] >= mid:
				H3 = cal
				break
		total_cal = H1 + H2 + H3
		if total_cal <= X:
			ans = mid
			low = mid + 1
		else:
			high = mid - 1
	print(ans)

if __name__ == "__main__":
	main()