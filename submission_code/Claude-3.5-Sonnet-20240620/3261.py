class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def can_achieve(target, k):
            mask = (1 << 30) - 1
            count = 0
            for num in nums:
                mask &= num
                if mask & target == mask:
                    mask = (1 << 30) - 1
                else:
                    count += 1
            return count <= k

        result = 0
        for bit in range(29, -1, -1):
            if not can_achieve(result | (1 << bit), k):
                result |= (1 << bit)
        return result