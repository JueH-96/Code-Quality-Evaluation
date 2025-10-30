class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0  # Need at least three elements to split into three parts
        
        mod = 10**9 + 7
        base = 911382629
        
        # Precompute prefix hashes H and power array P
        H = [0] * (n + 1)
        P = [1] * (n + 1)  # P[0] = 1, P[1] = base, etc.
        
        for i in range(n):
            H[i+1] = (H[i] * base + nums[i]) % mod
        
        for i in range(1, n+1):
            P[i] = (P[i-1] * base) % mod
        
        # Precompute first_ok array
        first_ok = [False] * n
        for i in range(n - 2):  # i can be up to n-3
            if 2 * i + 1 >= n:
                continue
            a = i + 1
            b = 2 * i + 1
            L = b - a + 1
            hash1 = H[i+1]
            hash2 = (H[b+1] - H[a] * P[L]) % mod
            if hash1 == hash2:
                first_ok[i] = True
        
        # Calculate first_count
        first_count = 0
        for i in range(n - 2):
            if first_ok[i]:
                min_j = 2 * i + 1
                if min_j > n - 2:
                    continue
                max_j = n - 2
                count = max_j - min_j + 1
                first_count += count
        
        # Calculate second_count
        second_count = 0
        for i in range(n - 2):
            max_j_second = min(n - 2, (n - 1 + i) // 2)
            for j in range(i + 1, max_j_second + 1):
                L = j - i
                end = j + L
                if end >= n:
                    continue
                # Compute hash2 (nums2) and hash3 (subarray in nums3)
                hash2 = (H[j+1] - H[i+1] * P[L]) % mod
                hash3 = (H[end+1] - H[j+1] * P[L]) % mod
                if hash2 == hash3:
                    # Check if not in first condition's region
                    if not (first_ok[i] and j >= 2 * i + 1):
                        second_count += 1
        
        return first_count + second_count