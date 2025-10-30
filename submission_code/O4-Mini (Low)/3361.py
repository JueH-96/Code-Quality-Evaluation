class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert to a list for mutability
        t = list(s)
        
        # Handle hour tens (t[0])
        if t[0] == '?':
            # If hour units is known and > '1', we cannot pick '1' (would make it >= 12)
            if t[1] != '?' and t[1] > '1':
                t[0] = '0'
            else:
                # Otherwise, '1' is the largest possible for the tens place
                t[0] = '1'
        
        # Handle hour units (t[1])
        if t[1] == '?':
            # If tens was set to '1', units can be at most '1'; if tens is '0', units can be '9'
            if t[0] == '1':
                t[1] = '1'
            else:
                t[1] = '9'
        
        # Handle minute tens (t[3])
        if t[3] == '?':
            # Tens of minutes can be 0..5; pick '5'
            t[3] = '5'
        
        # Handle minute units (t[4])
        if t[4] == '?':
            # Units of minutes can be 0..9; pick '9'
            t[4] = '9'
        
        return "".join(t)