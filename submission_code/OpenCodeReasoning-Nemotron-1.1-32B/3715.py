import bisect

class Solution:
	def maximumCoins(self, coins: List[List[int]], k: int) -> int:
		diff = {}
		for l, r, c in coins:
			diff[l] = diff.get(l, 0) + c
			diff[r+1] = diff.get(r+1, 0) - c
		
		events_list = sorted(diff.keys())
		cum_arr = []
		if events_list:
			cum = 0
			for pos in events_list:
				cum += diff[pos]
				cum_arr.append(cum)
		
		def get_coin(y):
			if not events_list or y < events_list[0]:
				return 0
			idx = bisect.bisect_right(events_list, y) - 1
			if idx < 0:
				return 0
			return cum_arr[idx]
		
		critical_x = set()
		for l, r, c in coins:
			critical_x.add(l)
			critical_x.add(r+1)
			critical_x.add(l - k)
			critical_x.add(r+1 - k)
		
		critical_x = sorted(critical_x)
		
		if not critical_x:
			return 0
		
		x0 = critical_x[0]
		total = 0
		for l, r, c in coins:
			L = max(l, x0)
			R = min(r, x0 + k - 1)
			if L <= R:
				total += c * (R - L + 1)
		
		current_f = total
		max_coins = current_f
		
		for i in range(1, len(critical_x)):
			x = critical_x[i]
			d = x - current_x
			s = get_coin(current_x + k) - get_coin(current_x)
			candidate1 = current_f
			candidate2 = current_f + (d - 1) * s
			max_in_interval = max(candidate1, candidate2)
			if max_in_interval > max_coins:
				max_coins = max_in_interval
			current_f = current_f + d * s
			current_x = x
		
		if current_f > max_coins:
			max_coins = current_f
		
		return max_coins