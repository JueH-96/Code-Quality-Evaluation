class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count_even = 0
        count_odd = 0
        best_even = 0
        best_odd = 0
        
        for num in nums:
            if num % 2 == 0:
                count_even += 1
                current_even = best_odd + 1
                if current_even > best_even:
                    best_even = current_even
            else:
                count_odd += 1
                current_odd = best_even + 1
                if current_odd > best_odd:
                    best_odd = current_odd
        
        candidate1 = max(count_even, count_odd)
        candidate2 = max(best_even, best_odd)
        return max(candidate1, candidate2)