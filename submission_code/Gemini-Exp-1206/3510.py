class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        maximumHeight.sort()
        total_sum = 0
        for i in range(n):
            if maximumHeight[i] < i + 1:
                return -1
            total_sum += maximumHeight[i]
        
        heights = []
        for i in range(n):
            heights.append(maximumHeight[i])
        
        heights.sort(reverse=True)
        
        ans = 0
        assigned = set()
        
        for h in heights:
            cur = h
            while cur > 0 and cur in assigned:
                cur -= 1
            if cur > 0:
                ans += cur
                assigned.add(cur)
        
        if len(assigned) != n:
            return -1
        
        return ans