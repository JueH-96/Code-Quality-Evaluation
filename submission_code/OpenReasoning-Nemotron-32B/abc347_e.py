import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	Q = int(next(it))
	queries = []
	if Q > 0:
		queries = [int(next(it)) for _ in range(Q)]
	
	events = [[] for _ in range(N+1)]
	size_arr = [0] * Q
	
	if Q > 0:
		present = [False] * (N+1)
		s = 0
		for i in range(Q):
			x = queries[i]
			if present[x]:
				s -= 1
				present[x] = False
			else:
				s += 1
				present[x] = True
			size_arr[i] = s
			events[x].append(i)
		
		prefix = [0] * Q
		prefix[0] = size_arr[0]
		for i in range(1, Q):
			prefix[i] = prefix[i-1] + size_arr[i]
	
	ans = [0] * (N+1)
	
	for j in range(1, N+1):
		lst = events[j]
		k = len(lst)
		for p in range(0, k, 2):
			start = lst[p]
			if p+1 < k:
				end = lst[p+1] - 1
			else:
				end = Q - 1
			if start <= end:
				if start == 0:
					total_val = prefix[end]
				else:
					total_val = prefix[end] - prefix[start-1]
				ans[j] += total_val
	
	res = []
	for i in range(1, N+1):
		res.append(str(ans[i]))
	print(" ".join(res))

if __name__ == '__main__':
	main()