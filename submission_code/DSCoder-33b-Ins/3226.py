class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        arr = []
        while nums:
            min_num = min(nums)
            nums.remove(min_num)
            arr.append(min_num)
            if nums:
                max_num = max(nums)
                nums.remove(max_num)
                arr.append(max_num)
        return arr