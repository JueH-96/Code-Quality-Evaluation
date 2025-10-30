class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        max_dist = 0
        
        count_N = count_S = count_E = count_W = 0
        
        for i in range(n):
            if s[i] == 'N':
                count_N += 1
            elif s[i] == 'S':
                count_S += 1
            elif s[i] == 'E':
                count_E += 1
            else:  # s[i] == 'W'
                count_W += 1
            
            x = count_E - count_W
            y = count_N - count_S
            
            # Maximize |x| + |y| by considering all four cases
            case1 = x + y + 2 * min(k, count_W + count_S)        # Maximize x + y
            case2 = x - y + 2 * min(k, count_W + count_N)        # Maximize x - y
            case3 = -x + y + 2 * min(k, count_E + count_S)       # Maximize -x + y
            case4 = -x - y + 2 * min(k, count_E + count_N)       # Maximize -x - y
            
            max_dist = max(max_dist, case1, case2, case3, case4)
        
        return max_dist