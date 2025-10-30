class Solution:
    def punishmentNumber(self, n: int) -> int:
        def canPartition(s, target, start=0):
            # Base case: if we've processed all characters
            if start == len(s):
                return target == 0
            
            # Try all possible substrings starting from 'start'
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                # Skip substrings with leading zeros (except single "0")
                if len(substring) > 1 and substring[0] == '0':
                    continue
                
                num = int(substring)
                # If this number is greater than remaining target, skip
                if num > target:
                    continue
                
                # Recursively check if remaining string can sum to remaining target
                if canPartition(s, target - num, end):
                    return True
            
            return False
        
        punishment = 0
        
        for i in range(1, n + 1):
            square = i * i
            square_str = str(square)
            
            # Check if square can be partitioned to sum to i
            if canPartition(square_str, i):
                punishment += square
        
        return punishment