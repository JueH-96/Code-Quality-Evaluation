import collections

class Solution:
    def get_square_free_part(self, num):
        square_free_part = 1
        i = 2
        while i * i <= num:
            if num % i == 0:
                count = 0
                while num % i == 0:
                    count += 1
                    num //= i
                if count % 2 == 1:
                    square_free_part *= i
            i += 1
        if num > 1:
            square_free_part *= num
        return square_free_part
        
    def maximumSum(self, nums: List[int]) -> int:
        square_free_parts = []
        for num in nums:
            square_free_parts.append(self.get_square_free_part(num))
        
        sf_to_indices = collections.defaultdict(list)
        for i in range(len(nums)):
            sf_part = square_free_parts[i]
            sf_to_indices[sf_part].append(i)
            
        max_sum = 0
        for sf_part in sf_to_indices:
            current_sum = 0
            indices = sf_to_indices[sf_part]
            for index in indices:
                current_sum += nums[index]
            max_sum = max(max_sum, current_sum)
            
        return max_sum