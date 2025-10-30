class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        prev_ands = {}
        ans = 0
        for num in nums:
            new_ands = {}
            for and_value in prev_ands:
                new_and = and_value & num
                new_ands[new_and] = new_ands.get(new_and, 0) + prev_ands[and_value]
            new_ands[num] = new_ands.get(num, 0) + 1
            if k in new_ands:
                ans += new_ands[k]
            prev_ands = new_ands
        return ans