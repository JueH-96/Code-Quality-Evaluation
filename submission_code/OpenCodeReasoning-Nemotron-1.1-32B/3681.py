from typing import List

class Solution:
	def maxRectangleArea(self, points: List[List[int]]) -> int:
		if len(points) < 4:
			return -1
		
		x_to_ys = {}
		for point in points:
			x, y = point
			if x not in x_to_ys:
				x_to_ys[x] = set()
			x_to_ys[x].add(y)
		
		xs = sorted(x_to_ys.keys())
		n = len(xs)
		max_area = -1
		
		for i in range(n):
			for j in range(i+1, n):
				x1 = xs[i]
				x2 = xs[j]
				common_ys = x_to_ys[x1] & x_to_ys[x2]
				if len(common_ys) < 2:
					continue
				common_ys_sorted = sorted(common_ys)
				for k in range(len(common_ys_sorted)):
					for l in range(k+1, len(common_ys_sorted)):
						y1 = common_ys_sorted[k]
						y2 = common_ys_sorted[l]
						corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
						valid = True
						for p in points:
							if (p[0], p[1]) in corners:
								continue
							if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
								valid = False
								break
						if valid:
							area = (x2 - x1) * (y2 - y1)
							if area > max_area:
								max_area = area
		return max_area