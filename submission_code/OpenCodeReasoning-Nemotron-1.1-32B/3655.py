import heapq

class Solution:
	def minOperations(self, n: int, m: int) -> int:
		d = len(str(n))
		max_val = 10**d - 1
		
		if max_val < 2:
			is_prime_arr = [False] * (max_val + 1)
		else:
			is_prime_arr = [True] * (max_val + 1)
			is_prime_arr[0] = False
			is_prime_arr[1] = False
			i = 2
			while i * i <= max_val:
				if is_prime_arr[i]:
					for j in range(i * i, max_val + 1, i):
						is_prime_arr[j] = False
				i += 1
		
		if is_prime_arr[n] or is_prime_arr[m]:
			return -1
		
		INF = 10**18
		dist = [INF] * (max_val + 1)
		heap = []
		heapq.heappush(heap, (n, n))
		dist[n] = n
		
		while heap:
			total, u = heapq.heappop(heap)
			if total != dist[u]:
				continue
			if u == m:
				return total
				
			s = str(u).zfill(d)
			for i in range(d):
				if s[i] != '9':
					new_digit = str(int(s[i]) + 1)
					new_s = s[:i] + new_digit + s[i+1:]
					v = int(new_s)
					if not is_prime_arr[v]:
						new_total = total + v
						if new_total < dist[v]:
							dist[v] = new_total
							heapq.heappush(heap, (new_total, v))
				if s[i] != '0':
					new_digit = str(int(s[i]) - 1)
					new_s = s[:i] + new_digit + s[i+1:]
					v = int(new_s)
					if not is_prime_arr[v]:
						new_total = total + v
						if new_total < dist[v]:
							dist[v] = new_total
							heapq.heappush(heap, (new_total, v))
							
		return -1