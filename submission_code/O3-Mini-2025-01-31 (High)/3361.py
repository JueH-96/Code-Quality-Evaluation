class Solution:
    def findLatestTime(self, s: str) -> str:
        max_time = ""
        # There are 12 valid hours (00 to 11) and 60 valid minutes (00 to 59)
        for h in range(12):
            for m in range(60):
                candidate = f"{h:02d}:{m:02d}"
                valid = True
                for i in range(5):
                    # Check each character: if s has a digit, it must match candidate's digit.
                    if s[i] != '?' and s[i] != candidate[i]:
                        valid = False
                        break
                if valid and candidate > max_time:
                    max_time = candidate
        return max_time