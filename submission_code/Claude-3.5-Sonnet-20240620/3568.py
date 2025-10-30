class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad the numbers with leading zeros to make them 4-digit strings
        nums = [f"{num:04d}" for num in (num1, num2, num3)]
        
        # Generate the key by taking the minimum digit at each position
        key = ''.join(min(num[i] for num in nums) for i in range(4))
        
        # Convert the key to an integer and return it
        return int(key)