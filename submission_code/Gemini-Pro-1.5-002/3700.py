class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        ans = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        for m in range(n):
                            if m == i or m == j or m == k or m == l:
                                continue
                            subsequence = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            counts = {}
                            for x in subsequence:
                                counts[x] = counts.get(x, 0) + 1
                            
                            mode_count = 0
                            mode = None
                            for x in counts:
                                if counts[x] > mode_count:
                                    mode_count = counts[x]
                                    mode = x
                                elif counts[x] == mode_count:
                                    mode = None
                                    break
                            if mode is not None and mode == subsequence[2]:
                                ans = (ans + 1) % MOD
        return ans