class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        count = 0
        suffix = int(s)
        if suffix < start or suffix > finish:
            return 0
        
        # Check if the suffix itself is within the range and valid
        if all(int(digit) <= limit for digit in s):
            count += 1
        
        # Generate numbers with s as suffix and check each digit
        prefix = 1
        while True:
            num = int(str(prefix) + s)
            if num > finish:
                break
            if num >= start:
                if all(int(digit) <= limit for digit in str(num)):
                    count += 1
            prefix += 1
        
        return count