import bisect
from typing import List

class Solution:
	def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
		L = 4 * side
		A = []
		for x, y in points:
			if y == 0:
				A.append(x)
			elif x == side:
				A.append(side + y)
			elif y == side:
				A.append(2 * side + (side - x))
			elif x == 0:
				A.append(3 * side + (side - y))
		A.sort()
		n = len(A)
		B = A + [x + L for x in A]
		
		def check(d):
			for i in range(n):
				count = 1
				current = B[i]
				next_index = i + 1
				for j in range(1, k):
					low_bound = current + d
					high_bound = current + (L - d)
					idx = bisect.bisect_left(B, low_bound, next_index, i + n)
					if idx == i + n:
						break
					if B[idx] > high_bound:
						break
					count += 1
					current = B[idx]
					next_index = idx + 1
				if count == k:
					gap = (B[i] + L) - current
					if d <= gap <= L - d:
						return True
			return False
		
		lo, hi = 0, 2 * side
		ans = 0
		while lo <= hi:
			mid = (lo + hi) // 2
			if check(mid):
				ans = mid
				lo = mid + 1
			else:
				hi = mid - 1
		return ans