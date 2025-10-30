class Solution:
    def makeStringGood(self, s: str) -> int:
        freq = [0] * 26
        for c in s:
            freq[ord(c) - ord('a')] += 1
        
        max_freq = max(freq) if any(freq) else 0
        min_cost = len(s)  # Initial cost is deleting all characters (t=0 case)
        upper_t = max_freq + 26
        
        for t in range(0, upper_t + 1):
            sum_not_in_S = 0
            sum_d = 0
            excess_S = 0
            deficit_S = 0
            
            for i in range(26):
                f = freq[i]
                cost_adjust = abs(f - t)
                if cost_adjust <= f:
                    sum_d += cost_adjust
                    if f > t:
                        excess_S += (f - t)
                    elif f < t:
                        deficit_S += (t - f)
                else:
                    sum_not_in_S += f
            
            current_cost = sum_not_in_S + sum_d - min(excess_S, deficit_S)
            if current_cost < min_cost:
                min_cost = current_cost
        
        return min_cost