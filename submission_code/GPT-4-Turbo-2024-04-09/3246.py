class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        # We need to find if there's a combination of two or more numbers
        # whose bitwise OR results in a number with at least one trailing zero.
        
        # A trailing zero in binary means that the least significant bit (rightmost) is zero.
        # To achieve this, we need at least one pair of numbers where in their OR operation,
        # at least one bit position results in a zero.
        
        # We can use a mask to check each bit position from 0 to 6 (since 100 in decimal is 1100100 in binary)
        # and see if we can find two numbers such that their OR results in a zero in that position.
        
        # Check all bit positions from 0 to 6 (since 1 <= nums[i] <= 100, max binary length needed is 7 bits)
        for bit_position in range(7):
            mask = 1 << bit_position
            all_have_bit_set = True
            for num in nums:
                if (num & mask) == 0:  # If any number has this bit not set, it's possible to get a zero here
                    all_have_bit_set = False
                    break
            if not all_have_bit_set:
                return True
        
        return False