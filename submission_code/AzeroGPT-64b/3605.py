class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for num in nums:
            # Find the smallest power of 2 that is greater than or equal to num
            # This is because a number and its next number can only form a power of 2
            # when they are the smallest possible numbers that satisfy the condition
            power_of_2 = 1
            while power_of_2 < num:
                power_of_2 *= 2
            
            # If the number is a power of 2 itself, there's no solution
            if power_of_2 == num:
                ans.append(-1)
            else:
                # The target number is formed by the smaller number and its next number
                # The smaller number is always power_of_2 / 2 or power_of_2 - 1
                # If power_of_2 / 2 plus 1 equals num, then it's the smaller number
                if power_of_2 // 2 + 1 == num:
                    ans.append(power_of_2 // 2)
                # Otherwise, the smaller number is power_of_2 - 1
                elif power_of_2 - 1 + 1 == num:
                    ans.append(power_of_2 - 1)
        return ans