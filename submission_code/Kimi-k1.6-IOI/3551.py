class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        max_m = n - 1
        f = [[-1] * (max_m + 1) for _ in range(n)]
        
        # Initialize base case for subarrays of length 1
        for l in range(n):
            f[l][0] = nums[l]
        
        # Compute f[l][m] for all valid l and m using dynamic programming
        for m in range(1, max_m + 1):
            for l in range(n - m):
                k = m.bit_length() - 1
                mask = 1 << k
                m_prime = m ^ mask
                f[l][m] = f[l][m_prime] ^ f[l + mask][m_prime]
        
        # Precompute prefix maximum arrays for each starting index l
        prefix_max = [[-1] * (max_m + 1) for _ in range(n)]
        for l in range(n):
            m_max = n - 1 - l
            current_max = f[l][0]
            prefix_max[l][0] = current_max
            for m in range(1, max_m + 1):
                if m <= m_max:
                    current_max = max(current_max, f[l][m])
                prefix_max[l][m] = current_max
        
        # Process each query to find the maximum XOR score in the specified range
        result = []
        for L, R in queries:
            max_xor = -1
            for l in range(L, R + 1):
                M = R - l
                if M < 0:
                    continue
                if M > max_m:
                    continue
                current = prefix_max[l][M]
                if current > max_xor:
                    max_xor = current
            result.append(max_xor)
        
        return result