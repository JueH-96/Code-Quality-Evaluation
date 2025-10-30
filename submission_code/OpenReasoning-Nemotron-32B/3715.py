import bisect

class Solution:
	def maximumCoins(self, coins: List[List[int]], k: int) -> int:
		critical_points = set()
		for l, r, c in coins:
			critical_points.add(l)
			critical_points.add(r + 1)
			critical_points.add(l - k)
			critical_points.add(r + 1 - k)
		critical_points = sorted(critical_points)
		
		if not critical_points:
			return 0
		
		coins_sorted = sorted(coins, key=lambda x: x[0])
		l_list = [coin[0] for coin in coins_sorted]
		
		def get_coin(p):
			idx = bisect.bisect_right(l_list, p) - 1
			if idx < 0:
				return 0
			seg = coins_sorted[idx]
			if p <= seg[1]:
				return seg[2]
			return 0
		
		x0 = critical_points[0]
		total = 0
		for l, r, c in coins:
			low = max(l, x0)
			high = min(r, x0 + k - 1)
			if low <= high:
				total += (high - low + 1) * c
		best = total
		current_x = x0
		current_f = total
		
		for i in range(len(critical_points) - 1):
			next_x = critical_points[i+1]
			d = next_x - current_x
			if d <= 0:
				continue
			s = get_coin(current_x + k) - get_coin(current_x)
			candidate1 = current_f
			candidate2 = current_f + (d - 1) * s
			if candidate1 > best:
				best = candidate1
			if candidate2 > best:
				best = candidate2
			current_f += d * s
			current_x = next_x
		
		if current_f > best:
			best = current_f
		
		return best