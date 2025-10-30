class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import Counter
        cnt = Counter(nums)
        for i in range(len(moveFrom)):
            if cnt[moveFrom[i]] > 0:
                cnt[moveFrom[i]] -= 1
                cnt[moveTo[i]] += 1
        return sorted(k for k, v in cnt.items() if v > 0)