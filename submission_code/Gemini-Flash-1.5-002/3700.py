class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        for m in range(l + 1, n):
                            subsequence = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            
                            counts = {}
                            for x in subsequence:
                                counts[x] = counts.get(x, 0) + 1
                            
                            max_count = 0
                            max_element = None
                            unique_mode = True
                            for x, c in counts.items():
                                if c > max_count:
                                    max_count = c
                                    max_element = x
                                    unique_mode = True
                                elif c == max_count and x != max_element:
                                    unique_mode = False
                            
                            if unique_mode and subsequence[2] == max_element:
                                count = (count + 1) % MOD
        return count