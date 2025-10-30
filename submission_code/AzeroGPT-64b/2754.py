class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        max_strength = float('-inf')

        # If the list contains only one element, return that element
        if n == 1:
            return nums[0]

        # If the list contains exactly two elements, return their product
        if n == 2:
            return nums[0] * nums[1]

        # Store the largest negative and smallest positive numbers for cases where there are no positive numbers at all
        largest_negative = float('-inf')
        smallest_positive = float('inf')

        positive_numbers = []
        negative_numbers = []

        # Split the list into positive and negative numbers and find the largest negative and smallest positive numbers
        for num in nums:
            if num > 0:
                positive_numbers.append(num)
                smallest_positive = min(smallest_positive, num)
            elif num < 0:
                negative_numbers.append(num)
                largest_negative = max(largest_negative, num)
        
        # There is at least one positive number or the only negative number
        if positive_numbers or len(negative_numbers) == 1:
            product = 1

            # Multiply all positive numbers
            for num in positive_numbers:
                product *= num
            
            # If there are an odd number of negative numbers, remove the largest (least negative)
            if len(negative_numbers) % 2:
                if positive_numbers:
                    max_strength = max(max_strength, product)
                else:
                    max_strength = max(max_strength, largest_negative)
                
            # Multiply all negative numbers if there is at least one positive number or if there is only one negative number
            else:
                for num in negative_numbers:
                    product *= num

                max_strength = max(max_strength, product)
        
        # If the list contains no positive numbers, find the greatest product among the combinations of the largest negative and the smallest positive
        else:
            max_strength = max(max_strength, largest_negative * smallest_positive)

        return max_strength