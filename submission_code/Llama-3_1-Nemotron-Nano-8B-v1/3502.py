class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = n * (n + 1) // 2
        if k == 0:
            return total
        count = [0] * 26
        left = 0
        max_count = 0
        bad = 0
        for right in range(n):
            c = ord(s[right]) - ord('a')
            count[c] += 1
            if count[c] > max_count:
                max_count = count[c]
            while max_count >= k:
                left_c = ord(s[left]) - ord('a')
                count[left_c] -= 1
                if count[left_c] + 1 == max_count:
                    has_other = False
                    for i in range(26):
                        if count[i] == max_count:
                            has_other = True
                            break
                    if not has_other:
                        max_count -= 1
                left += 1
            bad += right - left + 1
        return total - bad