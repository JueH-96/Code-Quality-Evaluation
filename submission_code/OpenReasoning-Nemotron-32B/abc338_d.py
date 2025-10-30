import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	X = list(map(int, data[2:2+m]))
	
	base = 0
	events = [0] * (n + 2)
	
	for i in range(m - 1):
		a1 = X[i]
		a2 = X[i + 1]
		low = min(a1, a2)
		high = max(a1, a2)
		L = high - low
		base += L
		v = n - 2 * L
		
		events[low] += v
		events[high] -= v
		
	current = 0
	ans = float('inf')
	for a in range(1, n + 1):
		current += events[a]
		total_here = base + current
		if total_here < ans:
			ans = total_here
			
	print(ans)

if __name__ == "__main__":
	main()