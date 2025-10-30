class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = collections.Counter(nums)
        result = 0
        for num in count:
            for i in range(1, 10):
                if num % 10 > i:
                    other_num = num % 100 // 10 * 10 + i
                    if other_num in count:
                        result += count[num] * count[other_num]
        return result