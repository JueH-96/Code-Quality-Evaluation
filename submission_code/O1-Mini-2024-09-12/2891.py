class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0

        max_num = max(nums)
        max_t = max_num + k
        timeline = [0] * (max_t + 2)

        for num in nums:
            start = max(0, num - k)
            end = num + k
            timeline[start] += 1
            timeline[end + 1] -= 1

        max_beauty = 0
        current = 0
        for count in timeline:
            current += count
            if current > max_beauty:
                max_beauty = current

        return max_beauty