class Solution:
    def findLatestTime(self, s: str) -> str:
        # We work with the time as characters for each position: s[0]s[1]:s[3]s[4]
        s = list(s)
        
        # Handle 'HH'
        # For the first hour digit, index 0:
        if s[0] == '?':
            # if second digit is '?' then we can choose '1' to get maximum "11"
            if s[1] == '?':
                s[0] = '1'
            else:
                # If s[1] is known, we need to decide based on validity.
                # Valid hours: 00 to 11.
                # If we choose '1' then hour becomes 1s[1]. This is valid only if s[1] is 0 or 1.
                # If s[1] > '1', then '1' would lead to an invalid hour (>= 12), so we choose '0'.
                if int(s[1]) > 1:
                    s[0] = '0'
                else:
                    s[0] = '1'
        
        # For the second hour digit, index 1:
        if s[1] == '?':
            # Now determine based on the first digit.
            if s[0] == '0':
                # hour is 0? and it must be <= 09 => maximum is '9'
                s[1] = '9'
            else:  # s[0] == '1'
                # The maximum hour is "11", so second digit must be at most '1'
                s[1] = '1'
        
        # Handle minutes
        # For minute tens (index 3) : maximum valid is 5.
        if s[3] == '?':
            s[3] = '5'
        
        # For minute ones (index 4) : maximum valid is 9.
        if s[4] == '?':
            s[4] = '9'
        
        return "".join(s)