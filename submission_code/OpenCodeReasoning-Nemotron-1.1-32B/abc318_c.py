import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	d = int(data[1])
	p = int(data[2])
	F = list(map(int, data[3:3+n]))
	
	total_sum = sum(F)
	F.sort(reverse=True)
	
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + F[i - 1]
	
	k_max = (n + d - 1) // d
	min_cost = float('inf')
	
	for k in range(0, k_max + 1):
		covered = min(k * d, n)
		cost = k * p + (total_sum - prefix[covered])
		if cost < min_cost:
			min_cost = cost
			
	print(min_cost)

if __name__ == '__main__':
	main()