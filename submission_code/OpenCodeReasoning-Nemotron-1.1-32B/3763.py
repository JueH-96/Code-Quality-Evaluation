from collections import defaultdict

class Solution:
	def separateSquares(self, squares: List[List[int]]) -> float:
		total_area = 0
		for sq in squares:
			l = sq[2]
			total_area += l * l
		
		target = total_area / 2.0
		
		events = defaultdict(int)
		for sq in squares:
			x, y_i, l_i = sq
			events[y_i] += l_i
			events[y_i + l_i] -= l_i
		
		ys = sorted(events.keys())
		current_slope = 0.0
		current_area = 0.0
		prev_y = None
		
		for y in ys:
			if prev_y is not None:
				if current_area >= target:
					return float(prev_y)
				
				if current_slope > 0:
					add_area = current_slope * (y - prev_y)
					if current_area + add_area >= target:
						return prev_y + (target - current_area) / current_slope
				
				current_area += current_slope * (y - prev_y)
			
			current_slope += events[y]
			prev_y = y
		
		if current_area >= target:
			return float(prev_y)
		
		return float(prev_y)