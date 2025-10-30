class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        total = 0
        curr = {}
        for a in nums:
            new_curr = {}
            new_curr[a] = new_curr.get(a, 0) + 1
            for val, cnt in curr.items():
                new_val = val & a
                new_curr[new_val] = new_curr.get(new_val, 0) + cnt
            curr = new_curr
            if k in curr:
                total += curr[k]
        return total