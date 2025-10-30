import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	queries = []
	index = 1+n+1
	for i in range(q):
		L = int(data[index])
		R = int(data[index+1])
		index += 2
		queries.append((L, R))
	
	next_arr = [n] * n
	for i in range(n):
		lo = i+1
		hi = n-1
		while lo <= hi:
			mid = (lo + hi) // 2
			if A[mid] >= 2 * A[i]:
				next_arr[i] = mid
				hi = mid - 1
			else:
				lo = mid + 1
				
	D = [next_arr[i] - i for i in range(n)]
	
	log_table = [0] * (n+1)
	for i in range(2, n+1):
		log_table[i] = log_table[i//2] + 1
		
	LOG = log_table[n] + 1 if n > 0 else 1
	st = [[0]*n for _ in range(LOG)]
	for i in range(n):
		st[0][i] = D[i]
		
	for j in range(1, LOG):
		step = 1 << (j-1)
		for i in range(0, n - (1<<j) + 1):
			st[j][i] = max(st[j-1][i], st[j-1][i+step])
			
	def query_max(l, r):
		if l > r:
			return 0
		length = r - l + 1
		k = log_table[length]
		return max(st[k][l], st[k][r - (1<<k) + 1])
	
	out_lines = []
	for (L, R) in queries:
		L0 = L - 1
		R0 = R - 1
		seg_len = R0 - L0 + 1
		low = 0
		high = seg_len // 2
		ans = 0
		while low <= high:
			mid = (low + high) // 2
			if mid == 0:
				ans = 0
				low = mid + 1
			else:
				M = query_max(L0, L0 + mid - 1)
				if M <= seg_len - mid:
					ans = mid
					low = mid + 1
				else:
					high = mid - 1
		out_lines.append(str(ans))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()