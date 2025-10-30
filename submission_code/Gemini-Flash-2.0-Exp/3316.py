class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        MOD = 10**9 + 7
        
        def get_power(subsequence):
            min_diff = float('inf')
            for i in range(len(subsequence)):
                for j in range(i + 1, len(subsequence)):
                    min_diff = min(min_diff, abs(subsequence[i] - subsequence[j]))
            return min_diff if min_diff != float('inf') else 0
        
        def find_subsequences(index, current_subsequence):
            if len(current_subsequence) == k:
                return [current_subsequence]
            
            if index == n:
                return []
            
            # Include the current element
            include_subsequences = find_subsequences(index + 1, current_subsequence + [nums[index]])
            
            # Exclude the current element
            exclude_subsequences = find_subsequences(index + 1, current_subsequence)
            
            return include_subsequences + exclude_subsequences
        
        all_subsequences = find_subsequences(0, [])
        
        valid_subsequences = [subsequence for subsequence in all_subsequences if len(subsequence) == k]
        
        total_power = 0
        for subsequence in valid_subsequences:
            total_power = (total_power + get_power(subsequence)) % MOD
        
        return total_power