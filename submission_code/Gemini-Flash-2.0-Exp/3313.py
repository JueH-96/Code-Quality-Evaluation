class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        def calculate_strength(subarrays):
            strength = 0
            for i, subarray in enumerate(subarrays):
                subarray_sum = sum(nums[subarray[0]:subarray[1] + 1])
                strength += ((-1)**i) * subarray_sum * (k - i)
            return strength

        max_strength = float('-inf')

        def find_subarrays(index, count, current_subarrays):
            nonlocal max_strength
            
            if count == k:
                max_strength = max(max_strength, calculate_strength(current_subarrays))
                return

            if index >= n:
                return

            for i in range(index, n):
                
                valid = True
                if current_subarrays:
                    if i <= current_subarrays[-1][1]:
                        valid = False
                
                if valid:
                    find_subarrays(i + 1, count + 1, current_subarrays + [(index, i)])
            
            

        find_subarrays(0, 0, [])

        return max_strength