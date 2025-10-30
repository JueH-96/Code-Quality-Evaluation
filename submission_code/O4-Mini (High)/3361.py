class Solution:
    def findLatestTime(self, s: str) -> str:
        # convert to a mutable list
        t = list(s)
        # hour tens
        if t[0] == '?':
            # if hour units is fixed > '1', we must pick '0'; otherwise pick '1'
            if t[1] != '?' and t[1] > '1':
                t[0] = '0'
            else:
                t[0] = '1'
        # hour units
        if t[1] == '?':
            # if tens is '0' we can go up to '9', else (tens '1') we can only go up to '1'
            t[1] = '9' if t[0] == '0' else '1'
        # minute tens
        if t[3] == '?':
            t[3] = '5'
        # minute units
        if t[4] == '?':
            t[4] = '9'
        return "".join(t)