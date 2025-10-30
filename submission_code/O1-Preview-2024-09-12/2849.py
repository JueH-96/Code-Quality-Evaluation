class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        total_imbalance = 0
        n = len(nums)
        for l in range(n):
            cnt = 0
            S = []
            S_set = set()
            for r in range(l, n):
                num = nums[r]
                if num in S_set:
                    total_imbalance += cnt
                    continue
                import bisect
                idx = bisect.bisect_left(S, num)
                delta = 0
                if idx > 0:
                    left = S[idx - 1]
                    if num - left > 1:
                        delta += 1
                if idx < len(S):
                    right = S[idx]
                    if right - num > 1:
                        delta += 1
                if idx > 0 and idx < len(S):
                    left = S[idx - 1]
                    right = S[idx]
                    if right - left > 1:
                        delta -= 1
                cnt += delta
                bisect.insort(S, num)
                S_set.add(num)
                total_imbalance += cnt
        return total_imbalance