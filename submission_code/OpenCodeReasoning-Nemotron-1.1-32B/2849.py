class Solution:
	def sumImbalanceNumbers(self, nums: List[int]) -> int:
		n = len(nums)
		positions_dict = {i: [] for i in range(1, n+1)}
		for i, num in enumerate(nums):
			if 1 <= num <= n:
				positions_dict[num].append(i)
		
		T = 0
		for x in range(1, n+1):
			if not positions_dict[x]:
				continue
			for y in range(x+2, n+1):
				if not positions_dict[y]:
					continue
				F_indices = []
				for z in range(x+1, y):
					if z in positions_dict and positions_dict[z]:
						F_indices.extend(positions_dict[z])
				F_indices.sort()
				
				segments = []
				if not F_indices:
					segments.append((0, n-1))
				else:
					if F_indices[0] > 0:
						segments.append((0, F_indices[0]-1))
					for i in range(1, len(F_indices)):
						if F_indices[i] > F_indices[i-1] + 1:
							segments.append((F_indices[i-1]+1, F_indices[i]-1))
					if F_indices[-1] < n-1:
						segments.append((F_indices[-1]+1, n-1))
				
				for (L, R) in segments:
					A = [i for i in positions_dict[x] if L <= i <= R]
					B = [i for i in positions_dict[y] if L <= i <= R]
					if not A or not B:
						continue
					total_seg = (R - L + 1) * (R - L + 2) // 2
					F_A_seg = self.compute_F_in_segment(A, L, R)
					F_B_seg = self.compute_F_in_segment(B, L, R)
					AB = sorted(set(A) | set(B))
					F_AB_seg = self.compute_F_in_segment(AB, L, R)
					count_seg = total_seg - F_A_seg - F_B_seg + F_AB_seg
					T += count_seg
		return T

	def compute_F_in_segment(self, positions, L, R):
		if not positions:
			total = (R - L + 1) * (R - L + 2) // 2
			return total
		total = 0
		start = L
		for pos in positions:
			if pos > start:
				seg_len = pos - start
				total += seg_len * (seg_len + 1) // 2
			start = pos + 1
		if start <= R:
			seg_len = R - start + 1
			total += seg_len * (seg_len + 1) // 2
		return total