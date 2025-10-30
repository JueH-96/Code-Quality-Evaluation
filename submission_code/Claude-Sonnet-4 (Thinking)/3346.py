class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def char_distance(c1, c2):
            pos1 = ord(c1) - ord('a')
            pos2 = ord(c2) - ord('a')
            clockwise = (pos2 - pos1) % 26
            counterclockwise = (pos1 - pos2) % 26
            return min(clockwise, counterclockwise)
        
        def smallest_reachable_char(c, budget):
            for i in range(26):
                target = chr(ord('a') + i)
                if char_distance(c, target) <= budget:
                    return target
        
        result = []
        remaining_budget = k
        
        for c in s:
            best_char = smallest_reachable_char(c, remaining_budget)
            distance_used = char_distance(c, best_char)
            result.append(best_char)
            remaining_budget -= distance_used
        
        return ''.join(result)