import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index])
		X = int(data[index+1])
		K = int(data[index+2])
		index += 3
		
		if N == 1:
			max_depth = 0
		else:
			max_depth = N.bit_length() - 1
		
		path = []
		cur = X
		while cur:
			path.append(cur)
			if cur == 1:
				break
			cur = cur // 2
		
		L = len(path)
		ans = 0
		for i in range(L):
			k_val = K - i
			if k_val < 0 or k_val > max_depth:
				continue
				
			if k_val == 0:
				ans += 1
			else:
				A = path[i]
				low = A << k_val
				if low > N:
					total = 0
				else:
					high = min(N, low + (1 << k_val) - 1)
					total = high - low + 1
				
				if i > 0:
					c = path[i-1]
					shift = k_val - 1
					low_b = c << shift
					if low_b > N:
						branch_total = 0
					else:
						high_b = min(N, low_b + (1 << shift) - 1)
						branch_total = high_b - low_b + 1
					ans += total - branch_total
				else:
					ans += total
					
		results.append(str(ans))
	
	print("
".join(results))

if __name__ == "__main__":
	main()