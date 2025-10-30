class Solution:
    def findLatestTime(self, s: str) -> str:
        # We'll parse s into a list for easier manipulation
        time = list(s)   # e.g., ["0","?",";", "5","?"]
        
        # time[0], time[1] -> hour
        # time[3], time[4] -> minute
        
        # 1) Fill hour tens (time[0])
        if time[0] == '?':
            # We try to set it to '1' if it doesn't force an invalid hour with time[1]
            if time[1] in ['?', '0', '1']:
                time[0] = '1'
            else:
                time[0] = '0'
        
        # 2) Fill hour ones (time[1])
        if time[1] == '?':
            # If hour tens is '1', the highest valid hour ones is '1' (hour=11 max)
            if time[0] == '1':
                time[1] = '1'
            else:
                # If hour tens is '0', we can go up to '9'
                time[1] = '9'
        
        # Now check if the formed hour is valid (must be <= 11)
        hour = int(time[0]) * 10 + int(time[1])
        if hour > 11:
            # That means we tried to put '1' as hour tens but hour turned out > 11.
            # So revert hour tens to '0' (largest valid hour tens if hour was invalid)
            time[0] = '0'
            # Re-check if time[1] was originally '?'. If so, we can set it to '9'
            # But if it was a digit, we keep it. If it invalidates the hour, 
            # the problem statement says there's guaranteed to be a valid time,
            # so this fix will make it valid.
            # If time[1] had been set from '?', we set it to '9'
            # If it was not '?', it should already be valid as a single digit with tens=0.
            # We'll do a final check:
            if s[1] == '?':
                time[1] = '9'
            # final hour check now
            hour = int(time[0]) * 10 + int(time[1])
            # Now it must be <= 11, given the problem guarantees a solution.
        
        # 3) Fill minute tens (time[3]) -> must be 0..5
        if time[3] == '?':
            time[3] = '5'  # pick the largest possible valid digit
        
        # 4) Fill minute ones (time[4]) -> must be 0..9
        if time[4] == '?':
            time[4] = '9'
        
        return "".join(time)