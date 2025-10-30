class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        def check_valid(arr1, arr2):
            # Check if arr1 is non-decreasing
            for i in range(1, n):
                if arr1[i] < arr1[i-1]:
                    return False
            
            # Check if arr2 is non-increasing
            for i in range(1, n):
                if arr2[i] > arr2[i-1]:
                    return False
            
            return True
        
        def generate_pairs(index, arr1, arr2):
            if index == n:
                if check_valid(arr1, arr2):
                    return 1
                return 0
                
            count = 0
            target = nums[index]
            
            # Try all possible combinations for current index
            for i in range(target + 1):
                arr1[index] = i
                arr2[index] = target - i
                
                # Early pruning - check if we can continue
                if index > 0:
                    if arr1[index] < arr1[index-1] or arr2[index] > arr2[index-1]:
                        continue
                        
                count = (count + generate_pairs(index + 1, arr1, arr2)) % MOD
                
            return count
        
        arr1 = [0] * n
        arr2 = [0] * n
        
        return generate_pairs(0, arr1, arr2)