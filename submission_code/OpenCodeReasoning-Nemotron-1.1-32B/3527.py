class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query_prefix(self, index):
		if index < 0:
			return 0
		i = index + 1
		s = 0
		while i > 0:
			s += self.tree[i]
			i -= i & -i
		return s

	def range_sum(self, l, r):
		if l > r:
			return 0
		return self.query_prefix(r) - self.query_prefix(l - 1)

class Solution:
	def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
		n = len(colors)
		arr = colors + colors
		size_diff = 2 * n - 1
		diff = [0] * size_diff
		for i in range(size_diff):
			if arr[i] != arr[i + 1]:
				diff[i] = 1
			else:
				diff[i] = 0
		
		fenw = Fenw(size_diff)
		for i in range(size_diff):
			fenw.update(i, diff[i])
		
		ans = []
		for query in queries:
			if query[0] == 1:
				k = query[1]
				count = 0
				for i in range(n):
					end_index = i + k - 2
					if end_index < size_diff:
						total = fenw.range_sum(i, end_index)
						if total == k - 1:
							count += 1
				ans.append(count)
			else:
				pos = query[1]
				new_color = query[2]
				if arr[pos] == new_color:
					pass
				else:
					arr[pos] = new_color
					arr[pos + n] = new_color
					indices = set()
					if pos >= 1:
						indices.add(pos - 1)
						indices.add(n + pos - 1)
					if pos < n - 1:
						indices.add(pos)
						indices.add(n + pos)
					if pos == 0 or pos == n - 1:
						indices.add(n - 1)
					for j in indices:
						new_val = 1 if arr[j] != arr[j + 1] else 0
						old_val = diff[j]
						if old_val != new_val:
							fenw.update(j, new_val - old_val)
							diff[j] = new_val
		return ans