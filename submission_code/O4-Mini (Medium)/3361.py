class Solution:
    def findLatestTime(self, s: str) -> str:
        # Convert to a mutable list of characters
        res = list(s)
        
        # Position 0: hour tens
        if res[0] == '?':
            # If the hour units is '?' or <= '1', we can put '1' here (to allow 10 or 11).
            # Otherwise we must put '0' (to keep hour <= 11).
            if res[1] == '?' or res[1] <= '1':
                res[0] = '1'
            else:
                res[0] = '0'
        
        # Position 1: hour units
        if res[1] == '?':
            # If we set tens to '1', units can be at most '1' (max hour = 11).
            # Otherwise (tens = '0'), units can go up to '9'.
            if res[0] == '1':
                res[1] = '1'
            else:
                res[1] = '9'
        
        # Position 3: minute tens
        if res[3] == '?':
            # Minutes tens place max is '5'
            res[3] = '5'
        
        # Position 4: minute units
        if res[4] == '?':
            # Minutes units place max is '9'
            res[4] = '9'
        
        return "".join(res)