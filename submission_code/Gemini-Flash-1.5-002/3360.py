class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter
        count = Counter(word)
        freqs = list(count.values())
        freqs.sort()
        n = len(freqs)
        
        def check(x):
            for i in range(n):
                if freqs[i] > x:
                    return False
            for i in range(n -1):
                if freqs[i+1] - freqs[i] > k:
                    return False
            return True

        left, right = 0, max(freqs)
        ans = float('inf')
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = min(ans, sum(max(0, freq - mid) for freq in freqs))
                right = mid -1
            else:
                left = mid + 1
        return ans