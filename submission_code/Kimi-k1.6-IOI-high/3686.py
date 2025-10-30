class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        # Rolling hash parameters
        base = 911382629  # Large prime base
        mod = 10**18 + 3  # Large modulus
        
        # Precompute power array where power[i] = base^i % mod
        power = [1] * (n + 1)
        for i in range(1, n + 1):
            power[i] = (power[i-1] * base) % mod
        
        # Compute prefix hashes H where H[i] is the hash of nums[0..i-1]
        H = [0] * (n + 1)
        for i in range(n):
            H[i+1] = (H[i] * base + nums[i]) % mod
        
        def get_hash(a, b):
            # Returns hash of nums[a..b] (0-based, inclusive)
            if a > b:
                return 0
            res = (H[b+1] - H[a] * power[b - a + 1]) % mod
            return res if res >= 0 else res + mod
        
        # Precompute if nums[0..i] == nums[i+1..2i+1]
        cond1_possible = [False] * n
        for i in range(n):
            end1 = 2 * i + 1
            if end1 >= n:
                continue
            h1 = get_hash(0, i)
            h2 = get_hash(i+1, end1)
            if h1 == h2:
                cond1_possible[i] = True
        
        count = 0
        
        for i in range(n - 2):
            max_j_cond2 = (n + i - 1) // 2
            for j in range(i + 1, n - 1):
                # Check Condition 1
                cond1 = cond1_possible[i] and j >= 2 * i + 1
                
                # Check Condition 2
                cond2 = False
                end2 = 2 * j - i
                if end2 < n and j <= max_j_cond2:
                    h2 = get_hash(i + 1, j)
                    h3_part = get_hash(j + 1, end2)
                    if h2 == h3_part:
                        cond2 = True
                
                if cond1 or cond2:
                    count += 1
        
        return count