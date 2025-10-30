class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        used = set()
        curr = 1
        total = 0
        count = 0
        
        while count < n:
            if curr not in used:
                complement = k - curr
                if complement not in used:
                    used.add(curr)
                    total += curr
                    count += 1
            curr += 1
            
        return total