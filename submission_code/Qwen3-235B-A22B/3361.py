class Solution:
    def findLatestTime(self, s: str) -> str:
        # Check all possible hours in descending order
        for h in reversed(range(12)):
            hh = f"{h:02d}"  # Format as two-digit string
            # Check if current hour matches the pattern of s's first two characters
            if (s[0] != '?' and s[0] != hh[0]) or (s[1] != '?' and s[1] != hh[1]):
                continue
            # Check all possible minutes in descending order
            for m in reversed(range(60)):
                mm = f"{m:02d}"  # Format as two-digit string
                # Check if the minute part matches the pattern of s's last two characters
                if (s[3] != '?' and s[3] != mm[0]) or (s[4] != '?' and s[4] != mm[1]):
                    continue
                # Return the first valid and latest time found
                return f"{hh}:{mm}"
        # According to problem constraints, there will always be at least one valid time
        return "00:00"