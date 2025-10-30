class Solution:
    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        # Precompute the divisors
        divisors = [num * k for num in nums2]
        
        count = 0
        # Iterate through each number in nums1
        for num1 in nums1:
            # Check against each precomputed divisor
            for divisor in divisors:
                if num1 % divisor == 0:
                    count += 1
        return count