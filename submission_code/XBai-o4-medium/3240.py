class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def total(num: int) -> int:
            res = 0
            m = x
            while True:
                val = 1 << (m - 1)
                if val > num:
                    break
                full = (num + 1) // (1 << m)
                res += full * (1 << (m - 1))
                rem = (num + 1) % (1 << m)
                res += max(0, rem - (1 << (m - 1)))
                m += x
            return res
        
        low = 0
        high = 10**18
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            try:
                t = total(mid)
            except:
                t = 10**19  # In case of overflow or error, assume a value larger than k
            if t <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans