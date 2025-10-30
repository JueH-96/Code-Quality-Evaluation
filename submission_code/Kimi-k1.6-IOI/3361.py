class Solution:
    def findLatestTime(self, s: str) -> str:
        # Generate all possible times in descending order (from 11:59 to 00:00)
        for hour in range(11, -1, -1):
            for minute in range(59, -1, -1):
                time_str = f"{hour:02d}:{minute:02d}"
                # Check if this time matches the pattern s
                match = True
                for i in [0, 1, 3, 4]:
                    if s[i] != '?' and s[i] != time_str[i]:
                        match = False
                        break
                if match:
                    return time_str
        return ""  # The problem guarantees at least one valid time, so this line is theoretically unreachable