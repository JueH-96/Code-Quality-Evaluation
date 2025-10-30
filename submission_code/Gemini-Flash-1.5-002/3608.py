import math

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        ans = 0
        for i in range(1 << n):
            seq1 = []
            indices1 = []
            for j in range(n):
                if (i >> j) & 1:
                    seq1.append(nums[j])
                    indices1.append(j)
            
            if not seq1:
                continue

            gcd1 = seq1[0]
            for x in seq1[1:]:
                gcd1 = math.gcd(gcd1, x)

            for k in range(1 << n):
                seq2 = []
                indices2 = []
                valid = True
                for l in range(n):
                    if (k >> l) & 1:
                        if l in indices1:
                            valid = False
                            break
                        seq2.append(nums[l])
                        indices2.append(l)
                
                if not valid or not seq2:
                    continue

                gcd2 = seq2[0]
                for x in seq2[1:]:
                    gcd2 = math.gcd(gcd2, x)

                if gcd1 == gcd2:
                    ans = (ans + 1) % MOD

        return ans