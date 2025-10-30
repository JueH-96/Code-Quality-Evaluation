class Solution:
    def findLatestTime(self, s: str) -> str:
        # Handle hours part
        hour_first = s[0]
        hour_second = s[1]
        
        if hour_first == '?':
            if hour_second == '?' or hour_second <= '1':
                hour_first_digit = '1'
            else:
                hour_first_digit = '0'
        else:
            hour_first_digit = hour_first
        
        if hour_second == '?':
            if hour_first_digit == '0':
                hour_second_digit = '9'
            elif hour_first_digit == '1':
                hour_second_digit = '1'
            else:
                hour_second_digit = '0'  # Fallback, should not happen per constraints
        else:
            hour_second_digit = hour_second
        
        # Handle minutes part
        min_first = s[3]
        min_second = s[4]
        
        if min_first == '?':
            min_first_digit = '5'
        else:
            min_first_digit = min_first
        
        if min_second == '?':
            min_second_digit = '9'
        else:
            min_second_digit = min_second
        
        # Build and return the result string
        return f"{hour_first_digit}{hour_second_digit}:{min_first_digit}{min_second_digit}"