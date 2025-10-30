class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(nums)
        result = []
        for i in range(0, len(sorted_nums), 2):
            # Append Bob's element (i+1) first, then Alice's (i)
            result.append(sorted_nums[i+1])
            result.append(sorted_nums[i])
        return result