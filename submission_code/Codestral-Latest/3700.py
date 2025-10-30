class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0

        for i in range(n - 4):
            for j in range(i + 1, n - 3):
                for k in range(j + 1, n - 2):
                    for l in range(k + 1, n - 1):
                        for m in range(l + 1, n):
                            seq = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            if seq[2] == max(seq, key=seq.count):
                                count = (count + 1) % MOD

        return count