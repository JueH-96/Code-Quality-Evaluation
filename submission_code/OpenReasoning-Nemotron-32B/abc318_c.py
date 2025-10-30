import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	D = int(data[1])
	P = int(data[2])
	F = list(map(int, data[3:3+n]))
	
	total = sum(F)
	F.sort(reverse=True)
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + F[i - 1]
		
	k_max = (n + D - 1) // D
	
	ans = float('inf')
	for k in range(0, k_max + 1):
		x = min(k * D, n)
		cost = k * P + total - prefix[x]
		if cost < ans:
			ans = cost
			
	print(ans)

if __name__ == '__main__':
	main()