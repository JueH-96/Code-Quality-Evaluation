class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        mod = 10**9 + 7

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        for m in range(l + 1, n):
                            subsequence = [nums[i], nums[j], nums[k], nums[l], nums[m]]
                            
                            # Check for unique middle mode
                            counts = {}
                            for num in subsequence:
                                counts[num] = counts.get(num, 0) + 1
                            
                            max_count = 0
                            mode = None
                            modes = []
                            
                            for num, c in counts.items():
                                if c > max_count:
                                    max_count = c
                                    modes = [num]
                                elif c == max_count:
                                    modes.append(num)
                            
                            if len(modes) == 1 and modes[0] == subsequence[2]:
                                count = (count + 1) % mod

        return count