class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        excluded = set()
        total = 0
        count = 0
        i = 1
        
        while count < n:
            if i not in excluded:
                total += i
                count += 1
                excluded.add(k - i)
            i += 1
        
        return total