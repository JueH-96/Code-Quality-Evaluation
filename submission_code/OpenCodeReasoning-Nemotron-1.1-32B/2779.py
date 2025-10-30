class Solution:
	def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
		arr = [0] * n
		total = 0
		ans = []
		for idx, c in queries:
			old = arr[idx]
			if old == c:
				ans.append(total)
				continue
				
			if old != 0:
				if idx - 1 >= 0 and arr[idx-1] == old:
					total -= 1
				if idx + 1 < n and arr[idx+1] == old:
					total -= 1
					
			arr[idx] = c
			
			if idx - 1 >= 0 and arr[idx-1] == c:
				total += 1
			if idx + 1 < n and arr[idx+1] == c:
				total += 1
				
			ans.append(total)
			
		return ans