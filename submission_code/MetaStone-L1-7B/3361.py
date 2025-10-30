class Solution:
    def findLatestTime(self, s: str) -> str:
        # Process the hour part to find the maximum possible hour
        max_h = 0
        for h in range(11, 0, -1):
            h_str = f"{h:02d}"
            possible = True
            for i in range(2):
                if s[i] != '?' and s[i] != h_str[i]:
                    possible = False
                    break
            if possible:
                max_h = h
                break
        
        # Process the minute part to find the maximum possible minute
        m_part = s[3:5]
        m1 = m_part[0]
        m2 = m_part[1]
        
        possible_m1 = []
        if m1 == '?':
            possible_m1 = ['0', '1', '2', '3', '4', '5']
        else:
            possible_m1 = [m1]
        
        possible_m2 = []
        if m2 == '?':
            possible_m2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            possible_m2 = [m2]
        
        max_minute = 0
        best_m1 = ''
        best_m2 = ''
        for m1c in possible_m1:
            for m2c in possible_m2:
                minute = int(m1c + m2c)
                if minute <= 59:
                    if minute > max_minute:
                        max_minute = minute
                        best_m1 = m1c
                        best_m2 = m2c
        
        # Construct the final time string
        time = f"{max_h:02d}:{best_m1}{best_m2}"
        return time