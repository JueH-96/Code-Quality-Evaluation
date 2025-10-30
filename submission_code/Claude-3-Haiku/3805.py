class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        s = '1' + s + '1'  # Augment the string with '1' at both ends
        
        max_active = 0
        active = 0
        trade_made = False
        
        for i in range(1, n+1):
            if s[i] == '1':
                active += 1
            else:
                if active > 0:
                    if not trade_made:
                        # Convert the block of '1's to '0's
                        max_active = max(max_active, active)
                        active = 0
                        trade_made = True
                    else:
                        # Convert the block of '0's to '1's
                        max_active = max(max_active, active + 1)
                        active = 0
                        trade_made = False
        
        return max_active