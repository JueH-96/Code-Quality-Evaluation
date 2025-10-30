class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        maximumHeight.sort(reverse=True)
        used = set()
        total_sum = 0
        
        for height in maximumHeight:
            for h in range(height, 0, -1):
                if h not in used:
                    used.add(h)
                    total_sum += h
                    break
            else:
                return -1
        
        return total_sum