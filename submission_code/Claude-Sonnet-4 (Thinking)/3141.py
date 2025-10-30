class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Create extended array (2 copies)
        extended = nums * 2
        
        # Dictionary to store minimum length for each sum
        sum_to_min_length = {}
        
        # Find all possible subarray sums and their minimum lengths
        for start in range(2 * n):
            current_sum = 0
            for end in range(start, 2 * n):
                current_sum += extended[end]
                length = end - start + 1
                
                # Only consider subarrays of length <= n
                if length > n:
                    break
                
                # Update minimum length for this sum
                if current_sum not in sum_to_min_length or length < sum_to_min_length[current_sum]:
                    sum_to_min_length[current_sum] = length
        
        result = float('inf')
        
        if total_sum > 0:
            # Try different values of k (number of complete copies)
            k = 0
            while True:
                required_sum = target - k * total_sum
                if required_sum <= 0:
                    if required_sum == 0 and k > 0:
                        result = min(result, k * n)
                    break
                if required_sum in sum_to_min_length:
                    total_length = sum_to_min_length[required_sum] + k * n
                    result = min(result, total_length)
                k += 1
        else:
            # If total_sum <= 0, we can only achieve target directly
            if target in sum_to_min_length:
                result = sum_to_min_length[target]
        
        return result if result != float('inf') else -1