class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        mod = 10**18 + 3
        base = 51
        n = len(nums)
        if n < 3:
            return 0
        
        # Precompute prefix hashes and powers of the base
        prefix_hash = [0] * (n + 1)
        powers = [1] * (n + 1)
        for i in range(n):
            prefix_hash[i+1] = (prefix_hash[i] * base + nums[i]) % mod
            powers[i+1] = (powers[i] * base) % mod
        
        def get_hash(a, b):
            if a > b:
                return 0
            return (prefix_hash[b+1] - prefix_hash[a] * powers[b - a + 1]) % mod
        
        count1 = 0
        max_i = (n - 2) // 2
        for i in range(max_i + 1):
            end_first = 2 * i + 1
            if end_first >= n:
                continue
            hash1 = get_hash(0, i)
            hash2 = get_hash(i+1, end_first)
            if hash1 == hash2:
                min_j = end_first
                max_j = n - 2
                if min_j > max_j:
                    continue
                count1 += (max_j - min_j + 1)
        
        count2 = 0
        for j in range(1, n - 1):
            max_k = min(j, (n - 1) - j)
            for k in range(1, max_k + 1):
                i = j - k
                if i < 0:
                    continue
                if j + k >= n:
                    continue
                hash_a = get_hash(i+1, j)
                hash_b = get_hash(j+1, j+1 + k - 1)
                if hash_a == hash_b:
                    count2 += 1
        
        count_both = 0
        for i in range(max_i + 1):
            end_first = 2 * i + 1
            if end_first >= n:
                continue
            hash1 = get_hash(0, i)
            hash2 = get_hash(i+1, end_first)
            if hash1 != hash2:
                continue
            j_min = end_first
            j_max_condition = ((n - 1) + i) // 2
            j_max = min(n - 2, j_max_condition)
            if j_min > j_max:
                continue
            for j in range(j_min, j_max + 1):
                k = j - i
                hash_a = get_hash(i+1, j)
                hash_b = get_hash(j+1, j+1 + k - 1)
                if hash_a == hash_b:
                    count_both += 1
        
        return (count1 + count2 - count_both) % mod