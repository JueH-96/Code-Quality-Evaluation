class Solution:
    def findLatestTime(self, s: str) -> str:
        def is_compatible(candidate, part):
            for i in range(2):
                if part[i] != '?' and part[i] != candidate[i]:
                    return False
            return True
        
        # Process hour part
        hour_part = s[:2]
        max_hour = "00"
        for h in range(11, -1, -1):
            candidate = f"{h:02d}"
            if is_compatible(candidate, hour_part):
                max_hour = candidate
                break
        
        # Process minute part
        minute_part = s[3:5]
        max_minute = "00"
        for m in range(59, -1, -1):
            candidate = f"{m:02d}"
            if is_compatible(candidate, minute_part):
                max_minute = candidate
                break
        
        return f"{max_hour}:{max_minute}"