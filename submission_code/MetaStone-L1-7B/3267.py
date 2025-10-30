class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        for l in range(n, 0, -1):
            for i in range(n - l + 1):
                substr = s[i:i+l]
                # Check if the substring is special (all characters same)
                if all(c == substr[0] for c in substr):
                    count = 0
                    for j in range(n - l + 1):
                        if s[j:j+l] == substr:
                            count += 1
                    if count >= 3:
                        return l
        return -1