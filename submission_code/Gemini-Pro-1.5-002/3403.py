class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        ans = 0
        i = 0
        while i < n:
            ans += 1
            counts = {}
            for j in range(i, n):
                counts[s[j]] = counts.get(s[j], 0) + 1
                
                balanced = True
                first_count = 0
                for count in counts.values():
                    if first_count == 0:
                        first_count = count
                    elif first_count != count:
                        balanced = False
                        break
                
                if balanced:
                    i = j + 1
                    break
        return ans