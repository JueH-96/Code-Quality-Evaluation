class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = "aeiou"
        n = len(s)
        ans = 0
        for i in range(n):
            v_count = 0
            c_count = 0
            for j in range(i, n):
                if s[j] in vowels:
                    v_count += 1
                else:
                    c_count += 1
                if v_count == c_count and (v_count * c_count) % k == 0:
                    ans += 1
        return ans