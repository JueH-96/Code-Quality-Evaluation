class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        nums_sorted = sorted(nums)
        target_sorted = sorted(target, reverse=True)
        total_operations = 0
        
        for t in target_sorted:
            min_increment = float('inf')
            best_index = -1
            for i in range(len(nums_sorted)):
                current_num = nums_sorted[i]
                if current_num >= t:
                    if current_num % t == 0:
                        increment = 0
                    else:
                        next_multiple = ((current_num // t) + 1) * t
                        increment = next_multiple - current_num
                else:
                    increment = t - current_num
                
                if increment < min_increment:
                    min_increment = increment
                    best_index = i
            
            if best_index != -1:
                total_operations += min_increment
                nums_sorted[best_index] += min_increment
        
        return total_operations