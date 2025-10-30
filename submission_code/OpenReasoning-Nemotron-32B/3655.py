import heapq

class Solution:
	def minOperations(self, n: int, m: int) -> int:
		max_num = 10000
		is_prime = [True] * max_num
		if max_num > 0:
			is_prime[0] = False
		if max_num > 1:
			is_prime[1] = False
		for i in range(2, int(max_num**0.5) + 1):
			if is_prime[i]:
				for j in range(i*i, max_num, i):
					is_prime[j] = False
		
		if is_prime[n] or is_prime[m]:
			return -1
		
		d = len(str(n))
		dist = [10**18] * max_num
		dist[n] = n
		pq = [(n, n)]
		
		while pq:
			cost, u = heapq.heappop(pq)
			if u == m:
				return cost
			if cost != dist[u]:
				continue
			s = str(u).zfill(d)
			for i in range(len(s)):
				digit = int(s[i])
				if digit < 9:
					new_digit = digit + 1
					new_s = s[:i] + str(new_digit) + s[i+1:]
					v = int(new_s)
					if v < max_num and not is_prime[v]:
						new_cost = cost + v
						if new_cost < dist[v]:
							dist[v] = new_cost
							heapq.heappush(pq, (new_cost, v))
				if digit > 0:
					new_digit = digit - 1
					new_s = s[:i] + str(new_digit) + s[i+1:]
					v = int(new_s)
					if v < max_num and not is_prime[v]:
						new_cost = cost + v
						if new_cost < dist[v]:
							dist[v] = new_cost
							heapq.heappush(pq, (new_cost, v))
		return -1