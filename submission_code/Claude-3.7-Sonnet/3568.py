class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad each number with leading zeros to ensure they are 4 digits
        str_num1 = str(num1).zfill(4)
        str_num2 = str(num2).zfill(4)
        str_num3 = str(num3).zfill(4)
        
        # Generate the key by taking the minimum digit at each position
        key = ""
        for i in range(4):
            min_digit = min(str_num1[i], str_num2[i], str_num3[i])
            key += min_digit
        
        # Convert the key back to an integer (this will automatically remove leading zeros)
        return int(key)