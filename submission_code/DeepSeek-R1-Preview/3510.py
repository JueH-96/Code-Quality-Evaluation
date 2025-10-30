class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        if n == 0:
            return -1
        
        sorted_max = sorted(maximumHeight)
        prev = sorted_max[-1]
        if prev < 1:
            return -1
        
        total = prev
        
        for i in range(len(sorted_max)-2, -1, -1):
            current_max = sorted_max[i]
            new_height = min(current_max, prev - 1)
            if new_height < 1:
                return -1
            total += new_height
            prev = new_height
        
        return total