class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_sum(num):
            res = 0
            i = x
            while True:
                half = 1 << (i - 1)
                if half > num:
                    break
                cycle = half << 1
                full_cycles = (num + 1) // cycle
                rem = (num + 1) % cycle
                cnt = full_cycles * half
                cnt += max(0, rem - half)
                res += cnt
                i += x
            return res
        
        low = 0
        high = 10**18
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            s = compute_sum(mid)
            if s <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans