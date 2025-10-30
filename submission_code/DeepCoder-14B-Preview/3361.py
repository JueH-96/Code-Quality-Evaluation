class Solution:
    def findLatestTime(self, s: str) -> str:
        for h in range(11, -1, -1):
            for m in range(59, -1, -1):
                time_str = f"{h:02}:{m:02}"
                match = True
                for i in range(5):
                    if i == 2:
                        continue
                    if s[i] != '?' and s[i] != time_str[i]:
                        match = False
                        break
                if match:
                    return time_str
        return ""