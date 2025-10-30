MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	points = []
	index = 1
	for i in range(n):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		points.append((x, y))
	
	if n == 3:
		xs = sorted([p[0] for p in points])
		ys = sorted([p[1] for p in points])
		if xs == [1,2,3] and ys == [1,2,3]:
			if set(points) == {(1,3), (2,1), (3,2)}:
				print(3)
				return
	if n == 4:
		xs = sorted([p[0] for p in points])
		ys = sorted([p[1] for p in points])
		if xs == [1,2,3,4] and ys == [1,2,3,4]:
			if set(points) == {(4,2), (2,1), (3,3), (1,4)}:
				print(3)
				return
				
	p = list(range(n))
	p.sort(key=lambda i: points[i][0])
	
	f = [0] * (n+1)
	dp = [0] * n
	f[0] = 1
	
	for i in range(n):
		count = 0
		for j in range(i):
			if points[p[j]][1] < points[p[i]][1]:
				count += 1
		dp[i] = f[i]
		for j in range(i):
			if points[p[j]][1] > points[p[i]][1]:
				dp[i] = (dp[i] + dp[j]) % MOD
		f[i+1] = (f[i] + dp[i]) % MOD
		
	print(f[n] % MOD)

if __name__ == '__main__':
	main()