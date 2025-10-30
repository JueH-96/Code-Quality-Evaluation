import bisect

class Solution:
    palindromes = None

    @staticmethod
    def generate_palindromes():
        max_num = 10**9 - 1
        palindromes = set()
        
        # Generate 1-digit palindromes
        for num in range(1, 10):
            palindromes.add(num)
        
        # Generate palindromes for lengths from 2 to 9 digits
        for digits in range(2, 10):
            if digits % 2 == 0:
                half = digits // 2
                start = 10 ** (half - 1)
                end = 10 ** half
                for first_half in range(start, end):
                    s = str(first_half)
                    p = int(s + s[::-1])
                    if p > max_num:
                        continue
                    palindromes.add(p)
            else:
                half = (digits + 1) // 2
                start = 10 ** (half - 1)
                end = 10 ** half
                for first_half in range(start, end):
                    s = str(first_half)
                    reversed_part = s[:-1][::-1]
                    p = int(s + reversed_part)
                    if p > max_num:
                        continue
                    palindromes.add(p)
        
        # Convert to a sorted list
        return sorted(palindromes)
    
    def minimumCost(self, nums: List[int]) -> int:
        if Solution.palindromes is None:
            Solution.palindromes = self.generate_palindromes()
        
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        min_cost = float('inf')
        for y in Solution.palindromes:
            idx = bisect.bisect_right(nums, y)
            cost = y * idx - prefix[idx] + (prefix[-1] - prefix[idx]) - y * (n - idx)
            if cost < min_cost:
                min_cost = cost
        
        return min_cost