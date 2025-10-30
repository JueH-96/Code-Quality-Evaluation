class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n - k
        
        def check(mask):
            count = 0
            current = -1  # starts with all bits set
            for num in nums:
                current &= num
                if (current & mask) == current:
                    count += 1
                    current = -1  # reset for next block
            return count >= m
        
        mask = (1 << 30) - 1  # start with all bits set
        for bit in reversed(range(30)):
            candidate_mask = mask & ~ (1 << bit)
            if check(candidate_mask):
                mask = candidate_mask
        return mask