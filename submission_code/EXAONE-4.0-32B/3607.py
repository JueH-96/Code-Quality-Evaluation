import bisect

class Solution:
    spf = None

    @classmethod
    def precompute_spf(cls, n):
        spf_arr = list(range(n+1))
        for i in range(2, int(n**0.5) + 1):
            if spf_arr[i] == i:
                for j in range(i*i, n+1, i):
                    if spf_arr[j] == j:
                        spf_arr[j] = i
        return spf_arr

    def minOperations(self, nums):
        if not nums:
            return 0
        max_val = 10**6
        if Solution.spf is None:
            Solution.spf = self.precompute_spf(max_val)
        
        chains = []
        for x in nums:
            chain = [x]
            current = x
            while current > 1 and Solution.spf[current] != current:
                current = current // Solution.spf[current]
                chain.append(current)
            chains.append(chain)
        
        n = len(nums)
        dp = []
        dp0 = [j for j in range(len(chains[0]))]
        dp.append(dp0)
        
        for i in range(1, n):
            prev_chain = chains[i-1]
            curr_chain = chains[i]
            m_prev = len(prev_chain)
            
            M = [0] * m_prev
            M[-1] = dp[i-1][-1]
            for k in range(m_prev-2, -1, -1):
                M[k] = min(dp[i-1][k], M[k+1])
            
            B = list(reversed(prev_chain))
            curr_dp = []
            for j in range(len(curr_chain)):
                val_j = curr_chain[j]
                pos = bisect.bisect_right(B, val_j) - 1
                if pos < 0:
                    curr_dp.append(float('inf'))
                else:
                    k0 = m_prev - 1 - pos
                    curr_dp.append(M[k0] + j)
            
            dp.append(curr_dp)
        
        ans = min(dp[-1])
        return ans if ans != float('inf') else -1