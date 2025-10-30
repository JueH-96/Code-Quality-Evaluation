import bisect

def horizontal_cuts(n, rectangles):
	if not rectangles:
		return False
	y1_list = []
	y2_list = []
	for rect in rectangles:
		x1, y1, x2, y2 = rect
		y1_list.append(y1)
		y2_list.append(y2)
	
	bottoms = sorted(set(y1_list))
	tops = sorted(set(y2_list))
	
	if not tops or not bottoms:
		return False
		
	arr_y1 = sorted(rectangles, key=lambda rect: rect[1])
	y1_sorted = [rect[1] for rect in arr_y1]
	max_y2_prefix = [0] * len(arr_y1)
	if len(arr_y1) > 0:
		max_y2_prefix[0] = arr_y1[0][3]
		for i in range(1, len(arr_y1)):
			max_y2_prefix[i] = max(max_y2_prefix[i-1], arr_y1[i][3])
	
	cond2_dict = {}
	for b in bottoms:
		idx = bisect.bisect_left(y1_sorted, b)
		if idx == 0:
			max_y2_val = -10**18
		else:
			max_y2_val = max_y2_prefix[idx-1]
		cond2_dict[b] = (max_y2_val <= b)
	
	cond2_arr = [cond2_dict[b] for b in bottoms]
	suffix_true = [False] * len(bottoms)
	if bottoms:
		suffix_true[-1] = cond2_arr[-1]
		for i in range(len(bottoms)-2, -1, -1):
			suffix_true[i] = cond2_arr[i] or suffix_true[i+1]
	
	arr_y1_desc = sorted(rectangles, key=lambda rect: rect[1], reverse=True)
	y1_sorted_desc = [rect[1] for rect in arr_y1_desc]
	min_y2_suffix = [0] * len(arr_y1_desc)
	if len(arr_y1_desc) > 0:
		min_y2_suffix[0] = arr_y1_desc[0][3]
		for i in range(1, len(arr_y1_desc)):
			min_y2_suffix[i] = min(min_y2_suffix[i-1], arr_y1_desc[i][3])
	
	for a in tops:
		idx = bisect.bisect_left(y1_sorted, a)
		if idx == 0:
			max_y2_val = -10**18
		else:
			max_y2_val = max_y2_prefix[idx-1]
		if not (max_y2_val <= a):
			continue
		
		idx_desc = bisect.bisect_left(y1_sorted_desc, a)
		if idx_desc < len(y1_sorted_desc):
			min_y2_R = min_y2_suffix[idx_desc]
		else:
			min_y2_R = 10**18
		
		if min_y2_R == 10**18:
			continue
			
		L = max(a + 1, min_y2_R)
		i = bisect.bisect_left(bottoms, L)
		if i < len(bottoms) and suffix_true[i]:
			return True
			
	return False

def vertical_cuts(n, rectangles):
	if not rectangles:
		return False
	x1_list = []
	x2_list = []
	for rect in rectangles:
		x1, y1, x2, y2 = rect
		x1_list.append(x1)
		x2_list.append(x2)
	
	lefts = sorted(set(x1_list))
	rights = sorted(set(x2_list))
	
	if not rights or not lefts:
		return False
		
	arr_x1 = sorted(rectangles, key=lambda rect: rect[0])
	x1_sorted = [rect[0] for rect in arr_x1]
	max_x2_prefix = [0] * len(arr_x1)
	if len(arr_x1) > 0:
		max_x2_prefix[0] = arr_x1[0][2]
		for i in range(1, len(arr_x1)):
			max_x2_prefix[i] = max(max_x2_prefix[i-1], arr_x1[i][2])
	
	cond2_dict = {}
	for b in lefts:
		idx = bisect.bisect_left(x1_sorted, b)
		if idx == 0:
			max_x2_val = -10**18
		else:
			max_x2_val = max_x2_prefix[idx-1]
		cond2_dict[b] = (max_x2_val <= b)
	
	cond2_arr = [cond2_dict[b] for b in lefts]
	suffix_true = [False] * len(lefts)
	if lefts:
		suffix_true[-1] = cond2_arr[-1]
		for i in range(len(lefts)-2, -1, -1):
			suffix_true[i] = cond2_arr[i] or suffix_true[i+1]
	
	arr_x1_desc = sorted(rectangles, key=lambda rect: rect[0], reverse=True)
	x1_sorted_desc = [rect[0] for rect in arr_x1_desc]
	min_x2_suffix = [0] * len(arr_x1_desc)
	if len(arr_x1_desc) > 0:
		min_x2_suffix[0] = arr_x1_desc[0][2]
		for i in range(1, len(arr_x1_desc)):
			min_x2_suffix[i] = min(min_x2_suffix[i-1], arr_x1_desc[i][2])
	
	for a in rights:
		idx = bisect.bisect_left(x1_sorted, a)
		if idx == 0:
			max_x2_val = -10**18
		else:
			max_x2_val = max_x2_prefix[idx-1]
		if not (max_x2_val <= a):
			continue
		
		idx_desc = bisect.bisect_left(x1_sorted_desc, a)
		if idx_desc < len(x1_sorted_desc):
			min_x2_R = min_x2_suffix[idx_desc]
		else:
			min_x2_R = 10**18
		
		if min_x2_R == 10**18:
			continue
			
		L = max(a + 1, min_x2_R)
		i = bisect.bisect_left(lefts, L)
		if i < len(lefts) and suffix_true[i]:
			return True
			
	return False

class Solution:
	def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
		if horizontal_cuts(n, rectangles) or vertical_cuts(n, rectangles):
			return True
		return False