class Solution:
	def numberOfAlternatingGroups(self, colors: List[int]) -> int:
		n = len(colors)
		count = 0
		for i in range(n):
			mid = (i + 1) % n
			left = colors[i]
			right = colors[(i + 2) % n]
			if colors[mid] != left and colors[mid] != right:
				count += 1
		return count