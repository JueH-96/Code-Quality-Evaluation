class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        ans = 0
        for i in range(1 << n):
            seq1 = []
            for j in range(n):
                if (i >> j) & 1:
                    seq1.append(nums[j])
            
            if not seq1:
                continue
            
            g1 = 0
            for x in seq1:
                g1 = gcd(g1, x) if g1 else x

            for k in range(1 << n):
                if i & k:
                    continue
                
                seq2 = []
                for j in range(n):
                    if (k >> j) & 1:
                        seq2.append(nums[j])

                if not seq2:
                    continue
                
                g2 = 0
                for x in seq2:
                    g2 = gcd(g2, x) if g2 else x
                
                if g1 == g2:
                    ans = (ans + 1) % MOD
        return ans