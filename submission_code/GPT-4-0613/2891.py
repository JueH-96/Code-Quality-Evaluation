class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        from collections import Counter
        count = Counter(nums)
        max_beauty = max(count.values())
        for num in count:
            if num - k in count:
                max_beauty = max(max_beauty, count[num] + count[num - k])
            if num + k in count:
                max_beauty = max(max_beauty, count[num] + count[num + k])
        return max_beauty