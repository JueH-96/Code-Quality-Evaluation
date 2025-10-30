class Solution:
    def generateKey(self, num1: int, num2: int, num3: int) -> int:
        # Pad numbers with leading zeros to make them 4 digits
        str1 = f"{num1:04d}"
        str2 = f"{num2:04d}"
        str3 = f"{num3:04d}"
        
        key_digits = []
        for a, b, c in zip(str1, str2, str3):
            min_digit = min(int(a), int(b), int(c))
            key_digits.append(str(min_digit))
        
        key_str = ''.join(key_digits)
        return int(key_str)