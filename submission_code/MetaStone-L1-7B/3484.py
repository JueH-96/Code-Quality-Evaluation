class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        candidates = [s]
        
        for i in range(len(s_list) - 1):
            a = s_list[i]
            b = s_list[i+1]
            if (int(a) % 2) == (int(b) % 2):
                # Swap the two adjacent characters
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                candidates.append(''.join(s_list))
                # Swap back to original state for the next iteration
                s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
        
        return min(candidates)