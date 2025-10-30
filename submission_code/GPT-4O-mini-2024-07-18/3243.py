class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        suffix_length = len(s)
        suffix_value = int(s)
        count = 0
        
        # Calculate the minimum number that can have s as a suffix
        min_powerful = suffix_value + (start // (10 ** suffix_length)) * (10 ** suffix_length)
        if min_powerful < start:
            min_powerful += 10 ** suffix_length
        
        # Calculate the maximum number that can have s as a suffix
        max_powerful = suffix_value + (finish // (10 ** suffix_length)) * (10 ** suffix_length)
        if max_powerful > finish:
            max_powerful -= 10 ** suffix_length
        
        # Iterate through all possible powerful integers
        for powerful in range(min_powerful, max_powerful + 1, 10 ** suffix_length):
            if powerful > finish:
                break
            if powerful >= start:
                # Check if all digits are <= limit
                if all(int(digit) <= limit for digit in str(powerful)):
                    count += 1
        
        return count