class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        res = 0
        prev = {}
        for num in nums:
            current = {}
            current[num] = current.get(num, 0) + 1
            for key, count in prev.items():
                and_val = key & num
                current[and_val] = current.get(and_val, 0) + count
            if k in current:
                res += current[k]
            prev = current
        return res