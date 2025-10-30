class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        count_N, count_E, count_S, count_W = 0, 0, 0, 0
        ans = 0
        
        for char in s:
            if char == 'N':
                count_N += 1
            elif char == 'E':
                count_E += 1
            elif char == 'S':
                count_S += 1
            elif char == 'W':
                count_W += 1
            
            base1 = (count_N + count_E) - (count_S + count_W)
            neg1 = count_S + count_W
            candidate1 = base1 + 2 * min(neg1, k)
            
            base2 = (count_E + count_S) - (count_N + count_W)
            neg2 = count_N + count_W
            candidate2 = base2 + 2 * min(neg2, k)
            
            base3 = (count_N + count_W) - (count_E + count_S)
            neg3 = count_E + count_S
            candidate3 = base3 + 2 * min(neg3, k)
            
            base4 = (count_S + count_W) - (count_N + count_E)
            neg4 = count_N + count_E
            candidate4 = base4 + 2 * min(neg4, k)
            
            current_max = max(candidate1, candidate2, candidate3, candidate4)
            if current_max > ans:
                ans = current_max
                
        return ans