class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        def traverse(curr, direc):
            while 0 <= curr < len(nums):
                if not nums[curr]: 
                    return curr
                curr += 1 if direc else -1
                nums[curr-1] -= 1
                direc = not direc
            return -1 # Should never hit "-1"
        
        valid_zips = set()
        for zippos in zip(nums, reversed(nums)):
            if 0 in zippos:
                valid_zips.add(zippos)
        for i, num in enumerate(nums):
            if num:
                l, r = traverse(i, True), traverse(i, False)
                if l == r != -1: return 0
                if (nums[l], nums[r]) not in valid_zips: return 0
        return sum(num == 0 for num in nums)