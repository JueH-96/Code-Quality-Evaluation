from collections import deque
from collections import defaultdict

class Solution:
	def minimumLength(self, s: str) -> int:
		n = len(s)
		if n <= 1:
			return n
		
		next_in_active = [None] * n
		prev_in_active = [None] * n
		removed = [False] * n
		
		occ = defaultdict(list)
		for i, char in enumerate(s):
			occ[char].append(i)
		
		for char, indices in occ.items():
			indices.sort()
			for j in range(len(indices)):
				if j > 0:
					prev_in_active[indices[j]] = indices[j-1]
				else:
					prev_in_active[indices[j]] = None
				if j < len(indices) - 1:
					next_in_active[indices[j]] = indices[j+1]
				else:
					next_in_active[indices[j]] = None
		
		q = deque()
		for i in range(n):
			if prev_in_active[i] is not None and next_in_active[i] is not None:
				q.append(i)
				
		while q:
			i = q.popleft()
			if removed[i]:
				continue
			if prev_in_active[i] is None or next_in_active[i] is None:
				continue
				
			left_index = prev_in_active[i]
			right_index = next_in_active[i]
			if removed[left_index] or removed[right_index]:
				continue
				
			removed[left_index] = True
			removed[right_index] = True
			
			L = prev_in_active[left_index]
			R = next_in_active[right_index]
			
			if L is not None:
				next_in_active[L] = i
			prev_in_active[i] = L
			
			if R is not None:
				prev_in_active[R] = i
			next_in_active[i] = R
			
			candidates = []
			if L is not None:
				candidates.append(L)
			if R is not None:
				candidates.append(R)
			candidates.append(i)
			
			for cand in candidates:
				if not removed[cand]:
					q.append(cand)
					
		count = 0
		for i in range(n):
			if not removed[i]:
				count += 1
				
		return count