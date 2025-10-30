class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        prev = {}
        for num in nums:
            current = {}
            current[num] = 1
            for and_val, count in prev.items():
                new_and = and_val & num
                if new_and in current:
                    current[new_and] += count
                else:
                    current[new_and] = count
            for and_val, count in current.items():
                if and_val == k:
                    total += count
            prev = current.copy()
        return total