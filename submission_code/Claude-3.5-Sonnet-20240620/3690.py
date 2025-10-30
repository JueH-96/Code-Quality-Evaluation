class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        def check(length):
            count = [0] * (n - length + 1)
            total = 0
            for i in range(n):
                if i >= length:
                    total -= count[i - length]
                if s[i] == '1':
                    total += 1
                    if i >= length - 1:
                        count[i - length + 1] = 1
                if total <= numOps or length - total <= numOps:
                    return True
            return False
        
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left