class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0
        
        # Iterate over all possible subsequences of length 5
        for i in range(n - 4):
            for j in range(i + 1, n - 3):
                for k in range(j + 1, n - 2):
                    for l in range(k + 1, n - 1):
                        for m in range(l + 1, n):
                            subseq = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            middle = subseq[2]
                            freq = {}
                            for num in subseq:
                                if num in freq:
                                    freq[num] += 1
                                else:
                                    freq[num] = 1
                            max_freq = max(freq.values())
                            modes = [num for num, f in freq.items() if f == max_freq]
                            if len(modes) == 1 and modes[0] == middle:
                                count = (count + 1) % MOD
        
        return count