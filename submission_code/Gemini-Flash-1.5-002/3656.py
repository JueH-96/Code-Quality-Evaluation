class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        while len(nums) > 0:
            seen = set()
            distinct = True
            for num in nums:
                if num in seen:
                    distinct = False
                    break
                seen.add(num)
            if distinct:
                break
            if len(nums) >=3:
                nums = nums[3:]
                count += 1
            else:
                nums = []
                count +=1
        return count