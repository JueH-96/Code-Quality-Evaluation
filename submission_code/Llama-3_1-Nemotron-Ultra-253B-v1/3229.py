import bisect

def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        candidates = set()
        
        # Add palindromic elements from nums
        for num in nums:
            if is_palindrome(num):
                candidates.add(num)
        
        # Sort the array and compute prefix sums
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + sorted_nums[i]
        
        # Determine median(s)
        medians = []
        if n % 2 == 1:
            medians.append(sorted_nums[n // 2])
        else:
            medians.append(sorted_nums[(n // 2) - 1])
            medians.append(sorted_nums[n // 2])
        
        # Generate candidates around medians
        for m in medians:
            for delta in range(-100, 101):
                candidate = m + delta
                if candidate <= 0:
                    continue
                if is_palindrome(candidate):
                    candidates.add(candidate)
        
        # Find the minimum cost
        min_cost = float('inf')
        for y in candidates:
            k = bisect.bisect_left(sorted_nums, y)
            sum_less = y * k - prefix[k]
            sum_greater = (prefix[n] - prefix[k]) - y * (n - k)
            total = sum_less + sum_greater
            if total < min_cost:
                min_cost = total
        
        return min_cost