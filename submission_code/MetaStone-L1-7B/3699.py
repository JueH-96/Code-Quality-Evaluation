class Solution:
    def numberOfSpecialSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [defaultdict(int) for _ in range(n + 1)]
        suffix = [defaultdict(int) for _ in range(n + 1)]
        
        # Populate prefix counts
        current = defaultdict(int)
        for i in range(n):
            current[nums[i]] += 1
            for k in current:
                prefix[i+1][k] = current[k]
        
        # Populate suffix counts
        current = defaultdict(int)
        for i in range(n-1, -1, -1):
            current[nums[i]] += 1
            for k in current:
                suffix[i][k] = current[k]
        
        count = 0
        
        # Iterate over all possible q and r with q < r
        for q in range(2, n):
            for r in range(q + 1, n):
                T = nums[q] * nums[r]
                # For p, we need nums[p] = T / nums[r], p < q
                if nums[r] == 0:
                    continue
                if T % nums[r] != 0:
                    continue
                required_p = T // nums[r]
                count_p = prefix[q].get(required_p, 0)
                
                # For s, we need nums[s] = T / nums[q], s > r
                if nums[q] == 0:
                    continue
                if T % nums[q] != 0:
                    continue
                required_s = T // nums[q]
                count_s = suffix[r+1].get(required_s, 0)
                
                count += count_p * count_s
        
        return count