class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        ans = 0
        remaining_k = k
        for p in range(29, -1, -1):
            max_consecutive = 0
            current_consecutive = 0
            for num in nums:
                if num & (1 << p):
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            if max_consecutive == 0:
                continue  # This bit is always 0, so it doesn't affect the OR
            operations_needed = max_consecutive - 1
            if operations_needed <= remaining_k:
                # Can set this bit to 0
                remaining_k -= operations_needed
            else:
                # Have to set this bit to 1
                ans |= (1 << p)
        return ans