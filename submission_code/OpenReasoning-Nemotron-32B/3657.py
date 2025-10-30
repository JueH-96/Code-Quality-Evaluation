import bisect
from typing import List

class Solution:
	def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
		if not rectangles:
			return False
		if self.horizontal_cuts(n, rectangles):
			return True
		if self.vertical_cuts(n, rectangles):
			return True
		return False

	def horizontal_cuts(self, n, rectangles):
		coords = set()
		for rect in rectangles:
			coords.add(rect[1])
			coords.add(rect[3])
		candidate_cut1_list = sorted(coords)
		
		rects_sorted_by_y1 = sorted(rectangles, key=lambda x: x[1])
		bottoms_all = [rect[1] for rect in rects_sorted_by_y1]
		prefix_max_top = [0] * len(rects_sorted_by_y1)
		if rects_sorted_by_y1:
			prefix_max_top[0] = rects_sorted_by_y1[0][3]
			for i in range(1, len(rects_sorted_by_y1)):
				prefix_max_top[i] = max(prefix_max_top[i-1], rects_sorted_by_y1[i][3])
		
		for cut1 in candidate_cut1_list:
			if cut1 <= 0 or cut1 >= n:
				continue
			idx = bisect.bisect_left(bottoms_all, cut1) - 1
			if idx >= 0:
				max_top_below = prefix_max_top[idx]
				if max_top_below > cut1:
					continue
			start_index = idx + 1
			if start_index >= len(rects_sorted_by_y1):
				continue
			R = rects_sorted_by_y1[start_index:]
			if len(R) < 2:
				continue
				
			coords_R = set()
			for rect in R:
				coords_R.add(rect[1])
				coords_R.add(rect[3])
			candidate_cut2_list = sorted(coords_R)
			
			min_bottom_R = R[0][1]
			max_bottom_R = R[-1][1]
			
			j = 0
			current_max = -1
			for cut2 in candidate_cut2_list:
				if cut2 <= max(cut1, min_bottom_R) or cut2 >= n:
					continue
				if cut2 > max_bottom_R:
					break
				while j < len(R) and R[j][1] < cut2:
					current_max = max(current_max, R[j][3])
					j += 1
				if current_max <= cut2:
					return True
		return False

	def vertical_cuts(self, n, rectangles):
		coords = set()
		for rect in rectangles:
			coords.add(rect[0])
			coords.add(rect[2])
		candidate_cut1_list = sorted(coords)
		
		rects_sorted_by_x1 = sorted(rectangles, key=lambda x: x[0])
		lefts_all = [rect[0] for rect in rects_sorted_by_x1]
		prefix_max_right = [0] * len(rects_sorted_by_x1)
		if rects_sorted_by_x1:
			prefix_max_right[0] = rects_sorted_by_x1[0][2]
			for i in range(1, len(rects_sorted_by_x1)):
				prefix_max_right[i] = max(prefix_max_right[i-1], rects_sorted_by_x1[i][2])
				
		for cut1 in candidate_cut1_list:
			if cut1 <= 0 or cut1 >= n:
				continue
			idx = bisect.bisect_left(lefts_all, cut1) - 1
			if idx >= 0:
				max_right_below = prefix_max_right[idx]
				if max_right_below > cut1:
					continue
			start_index = idx + 1
			if start_index >= len(rects_sorted_by_x1):
				continue
			R = rects_sorted_by_x1[start_index:]
			if len(R) < 2:
				continue
				
			coords_R = set()
			for rect in R:
				coords_R.add(rect[0])
				coords_R.add(rect[2])
			candidate_cut2_list = sorted(coords_R)
			
			min_left_R = R[0][0]
			max_left_R = R[-1][0]
			
			j = 0
			current_max = -1
			for cut2 in candidate_cut2_list:
				if cut2 <= max(cut1, min_left_R) or cut2 >= n:
					continue
				if cut2 > max_left_R:
					break
				while j < len(R) and R[j][0] < cut2:
					current_max = max(current_max, R[j][2])
					j += 1
				if current_max <= cut2:
					return True
		return False