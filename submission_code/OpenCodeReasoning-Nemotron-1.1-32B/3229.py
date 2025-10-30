from typing import List

class Solution:
	def minimumCost(self, nums: List[int]) -> int:
		def is_palindrome(x):
			s = str(x)
			return s == s[::-1]
		
		def form_palindrome(left, n):
			if n % 2 == 0:
				res = left + left[::-1]
			else:
				res = left + left[:-1][::-1]
			return int(res)
		
		def next_palindrome(x):
			if x >= 10**9 - 1:
				return None
			s = str(x)
			n = len(s)
			left = s[:(n+1)//2]
			candidate = form_palindrome(left, n)
			if candidate > x:
				return candidate
			new_left = str(int(left) + 1)
			if len(new_left) > len(left):
				if n + 1 > 9:
					return None
				return 10**n + 1
			else:
				return form_palindrome(new_left, n)
		
		def prev_palindrome(x):
			if x <= 1:
				return None
			s = str(x)
			n = len(s)
			left = s[:(n+1)//2]
			candidate = form_palindrome(left, n)
			if candidate < x:
				return candidate
			new_left = str(int(left) - 1)
			if new_left == '0':
				if n - 1 == 0:
					return None
				return 10**(n-1) - 1
			if len(new_left) < len(left):
				if n - 1 == 0:
					return None
				return 10**(n-1) - 1
			else:
				return form_palindrome(new_left, n)
		
		n = len(nums)
		nums.sort()
		med = nums[n//2]
		candidates = set()
		if is_palindrome(med):
			candidates.add(med)
		else:
			cand1 = prev_palindrome(med)
			cand2 = next_palindrome(med)
			if cand1 is not None:
				candidates.add(cand1)
			if cand2 is not None:
				candidates.add(cand2)
		
		min_cost = float('inf')
		for cand in candidates:
			total = 0
			for num in nums:
				total += abs(num - cand)
			if total < min_cost:
				min_cost = total
		return min_cost