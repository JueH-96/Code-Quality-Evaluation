class Solution:
    def findLatestTime(self, s: str) -> str:
        # Iterate through all possible times in descending order
        for hour in range(11, -1, -1):
            for minute in range(59, -1, -1):
                # Format hour and minute as two-digit strings
                time_str = f"{hour:02d}:{minute:02d}"
                # Check each character in the time string against the input
                valid = True
                for i in [0, 1, 3, 4]:
                    if s[i] != '?' and s[i] != time_str[i]:
                        valid = False
                        break
                if valid:
                    return time_str
        return ""  # The problem guarantees at least one valid time, so this line is a fallback