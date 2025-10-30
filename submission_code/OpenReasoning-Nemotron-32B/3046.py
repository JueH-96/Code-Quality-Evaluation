class Solution:
	def minimumOperations(self, num: str) -> int:
		n = len(num)
		min_del = n
		
		patterns = [("2","5"), ("5","0"), ("7","5"), ("0","0")]
		
		for c1, c2 in patterns:
			list1 = []
			list2 = []
			for idx, digit in enumerate(num):
				if digit == c1:
					list1.append(idx)
				if digit == c2:
					list2.append(idx)
			for j in list2:
				candidate_i = -1
				for i in list1:
					if i < j:
						if i > candidate_i:
							candidate_i = i
				if candidate_i != -1:
					min_del = min(min_del, n - (candidate_i + 2))
		
		if '0' in num:
			min_del = min(min_del, n - 1)
		
		return min_del