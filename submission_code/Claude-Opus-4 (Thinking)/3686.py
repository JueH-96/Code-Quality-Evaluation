class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Precompute longest common prefix starting at positions i and j
        lcp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if nums[i] == nums[j]:
                    if i + 1 < n and j + 1 < n:
                        lcp[i][j] = 1 + lcp[i + 1][j + 1]
                    else:
                        lcp[i][j] = 1
        
        count = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Check if nums[0:i] is a prefix of nums[i:j]
                is_prefix1 = i <= j - i and lcp[0][i] >= i
                
                # Check if nums[i:j] is a prefix of nums[j:]
                is_prefix2 = j - i <= n - j and lcp[i][j] >= j - i
                
                if is_prefix1 or is_prefix2:
                    count += 1
        
        return count