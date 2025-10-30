import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	k = int(next(it))
	X = [int(next(it)) for _ in range(n)]
	A = [int(next(it)) for _ in range(n)]
	
	if k == 0:
		print(" ".join(map(str, A)))
		return
		
	f = [x - 1 for x in X]
	layers = 61
	dp = [[0] * n for _ in range(layers)]
	dp[0] = f
	
	for j in range(1, layers):
		prev_layer = dp[j-1]
		current_layer = dp[j]
		for i in range(n):
			current_layer[i] = prev_layer[prev_layer[i]]
			
	result_arr = []
	for i in range(n):
		current = i
		for bit in range(layers):
			if (k >> bit) & 1:
				current = dp[bit][current]
		result_arr.append(str(A[current]))
		
	print(" ".join(result_arr))

if __name__ == '__main__':
	main()