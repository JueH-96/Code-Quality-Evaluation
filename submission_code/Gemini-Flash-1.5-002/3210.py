class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        count = 0
        vowels = "aeiou"
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                v_count = 0
                c_count = 0
                for char in substring:
                    if char in vowels:
                        v_count += 1
                    else:
                        c_count += 1
                if v_count == c_count and (v_count * c_count) % k == 0:
                    count += 1
        return count