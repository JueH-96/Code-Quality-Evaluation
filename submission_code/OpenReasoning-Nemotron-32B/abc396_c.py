import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return

	n = int(data[0])
	m = int(data[1])
	black = list(map(int, data[2:2+n]))
	white = list(map(int, data[2+n:2+n+m]))
	
	black.sort(reverse=True)
	white.sort(reverse=True)
	
	prefixB = [0] * (n+1)
	for i in range(1, n+1):
		prefixB[i] = prefixB[i-1] + black[i-1]
	
	prefixW = [0] * (m+1)
	for j in range(1, m+1):
		prefixW[j] = prefixW[j-1] + white[j-1]
	
	maxB = [0] * (n+1)
	if n > 0:
		maxB[n] = prefixB[n]
		for i in range(n-1, -1, -1):
			maxB[i] = max(prefixB[i], maxB[i+1])
	else:
		maxB[0] = prefixB[0]
	
	T_max = min(n, m)
	ans = -10**18
	for t in range(0, T_max+1):
		candidate = prefixW[t] + maxB[t]
		if candidate > ans:
			ans = candidate
			
	print(ans)

if __name__ == '__main__':
	main()