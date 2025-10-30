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
	
	maxA = [0] * (n+1)
	if n > 0:
		maxA[n] = prefixB[n]
		for i in range(n-1, -1, -1):
			maxA[i] = max(prefixB[i], maxA[i+1])
	else:
		maxA[0] = prefixB[0]
	
	prefixW = [0] * (m+1)
	for i in range(1, m+1):
		prefixW[i] = prefixW[i-1] + white[i-1]
	
	ans = 0
	end_k = min(n, m)
	for k in range(0, end_k+1):
		total = maxA[k] + prefixW[k]
		if total > ans:
			ans = total
			
	print(ans)

if __name__ == '__main__':
	main()