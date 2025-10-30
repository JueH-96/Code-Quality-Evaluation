class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        result = 0
        
        # For each possible middle position
        for mid in range(2, n - 2):
            v = nums[mid]  # Middle value
            
            # Try all combinations of 2 elements before and 2 after
            for i1 in range(mid):
                for i2 in range(i1 + 1, mid):
                    for i3 in range(mid + 1, n):
                        for i4 in range(i3 + 1, n):
                            # Build frequency map
                            freq = {}
                            for idx in [i1, i2, mid, i3, i4]:
                                freq[nums[idx]] = freq.get(nums[idx], 0) + 1
                            
                            # Check if v is unique mode
                            v_freq = freq[v]
                            is_unique = True
                            for val, f in freq.items():
                                if val != v and f >= v_freq:
                                    is_unique = False
                                    break
                            
                            if is_unique:
                                result = (result + 1) % MOD
        
        return result