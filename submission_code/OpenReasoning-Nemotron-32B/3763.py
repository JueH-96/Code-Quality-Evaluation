from collections import defaultdict
from typing import List

class Solution:
	def separateSquares(self, squares: List[List[int]]) -> float:
		total_area = 0
		events = defaultdict(list)
		critical_points = set()
		critical_points.add(0)
		
		for square in squares:
			x, y, l = square
			total_area += l * l
			critical_points.add(y)
			critical_points.add(y + l)
			events[y].append(l)
			events[y + l].append(-l)
		
		critical_points = sorted(critical_points)
		target = total_area / 2.0
		
		current_slope = 0
		current_area = 0.0
		
		for i in range(len(critical_points) - 1):
			y0 = critical_points[i]
			if y0 in events:
				for change in events[y0]:
					current_slope += change
			
			y1 = critical_points[i+1]
			seg_length = y1 - y0
			seg_area = current_slope * seg_length
			
			if current_area <= target <= current_area + seg_area:
				if current_slope == 0:
					return y0
				else:
					h = y0 + (target - current_area) / current_slope
					return h
			
			current_area += seg_area
		
		return critical_points[-1]