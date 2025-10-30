class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes_to_00_11 = 0
        changes_to_01_10 = 0
        
        # We will consider pairs since the string length is even
        for i in range(0, n, 2):
            pair = s[i:i+2]
            if pair in {"00", "11"}:
                changes_to_01_10 += 1
            elif pair in {"01", "10"}:
                changes_to_00_11 += 1
            else:
                # For any other case, which is not possible in a binary string
                continue
        
        # Minimum changes needed to make all pairs either "00" or "11" or "01" or "10"
        return min(changes_to_00_11, changes_to_01_10)