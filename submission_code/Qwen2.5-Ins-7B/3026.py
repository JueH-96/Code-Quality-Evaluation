class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        if n == 1:
            return 1
        
        sum_val = 0
        used = set()
        i = 1
        
        while len(used) < n:
            if target - i not in used:
                sum_val += i
                used.add(i)
            i += 1
        
        return sum_val % (10**9 + 7)