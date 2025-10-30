class Solution:
	def sumImbalanceNumbers(self, nums: List[int]) -> int:
		n = len(nums)
		total_ans = 0
		MAX_VAL = 1000
		for l in range(n):
			active = [False] * (MAX_VAL + 1)
			count = 0
			for r in range(l, n):
				x = nums[r]
				if not active[x]:
					active[x] = True
					pred = -1
					for i in range(x - 1, 0, -1):
						if active[i]:
							pred = i
							break
					succ = -1
					for i in range(x + 1, MAX_VAL + 1):
						if active[i]:
							succ = i
							break
					if pred != -1 and succ != -1:
						if succ - pred > 1:
							count -= 1
					if pred != -1:
						if x - pred > 1:
							count += 1
					if succ != -1:
						if succ - x > 1:
							count += 1
				total_ans += count
		return total_ans