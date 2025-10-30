class Solution:
    def minEnd(self, n: int, x: int) -> int:
        def count(y, x, bit=31):
            if bit < 0:
                return 1
            mask = 1 << bit
            if x & mask:
                if y & mask:
                    return count(y, x, bit - 1)
                else:
                    return 0
            else:
                if y & mask:
                    return 2 * count(y, x, bit - 1)
                else:
                    return count(y, x, bit - 1)
        
        left = x
        right = x + n * 2  # Upper bound estimate
        answer = right
        while left <= right:
            mid = (left + right) // 2
            cnt = count(mid, x)
            if cnt >= n:
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer