class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # We will iterate over each bit position from 0 to 31 (enough for 32-bit integers).
        # For each bit position i, count how many numbers in nums have bit i set.
        # If at least k elements have bit i set, set bit i in the result.
        
        result = 0
        for i in range(32):
            count = 0
            for n in nums:
                if n & (1 << i):
                    count += 1
            if count >= k:
                result |= (1 << i)
        return result