class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        last = -10**9
        i = 0
        n = len(nums)
        while i < n:
            j = i
            current_bit = bin(nums[i]).count("1")
            seg = []
            while j < n and bin(nums[j]).count("1") == current_bit:
                seg.append(nums[j])
                j += 1
            seg.sort()
            if seg[0] < last:
                return False
            last = seg[-1]
            i = j
        return True