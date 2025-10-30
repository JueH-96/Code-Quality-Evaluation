class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        arr = [0] * (len(s) + 1)
        for i in range(1, len(s) + 1):
            arr[i] = arr[i - 1] + (s[i - 1] == '1')

        ans, cntZero = 0, 0
        for i in range(len(s)):
            if s[i] == '0':
                cntZero += 1
                lo = bisect_left(arr, arr[i] - cntZero * cntZero)
                hi = bisect_right(arr, arr[i] - (cntZero + 1) * (cntZero + 1) - 1)
                ans += hi - lo
        return ans