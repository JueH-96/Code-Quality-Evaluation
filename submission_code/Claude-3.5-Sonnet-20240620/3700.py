class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0
        
        for i in range(2, n-2):
            left = set(nums[:i])
            right = set(nums[i+1:])
            
            if nums[i] not in left and nums[i] not in right:
                left_count = len(left)
                right_count = len(right)
                count += (
                    self.combination(i, 2) * 
                    self.combination(n-i-1, 2)
                ) % MOD
        
        return count % MOD
    
    def combination(self, n: int, k: int) -> int:
        if k > n:
            return 0
        result = 1
        for i in range(k):
            result *= (n - i)
            result //= (i + 1)
        return result