class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        a = [0] + coins
        def count_less_equal(val):
            ans = 0
            cnt = [val//x+1 for x in a]
            for mask in range(1, 1<<n):
                bitcnt = mask.bit_count()
                base = 1
                for i in range(n):
                    if mask >> i & 1:
                        base *= cnt[i+1]
                ans += (-1)**bitcnt * (base-1)
            return ans
        
        l, r = 0, 2000 * k
        while l < r:
            mid = l + r >> 1
            if count_less_equal(mid) < k:
                l = mid + 1
            else:
                r = mid
        return l