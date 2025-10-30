class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        count_e = count_w = count_n = count_s = 0
        ans = 0
        for i, ch in enumerate(s):
            if ch == 'E':
                count_e += 1
            elif ch == 'W':
                count_w += 1
            elif ch == 'N':
                count_n += 1
            else:
                count_s += 1
            a = min(count_e, count_w)
            b = min(count_n, count_s)
            total = i + 1
            if k >= a + b:
                candidate = total
            else:
                candidate = total - 2 * (a + b) + 2 * k
            if candidate > ans:
                ans = candidate
        return ans