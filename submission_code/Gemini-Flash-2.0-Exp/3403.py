class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        count = 0
        i = 0
        while i < n:
            count += 1
            j = i + 1
            while j <= n:
                sub = s[i:j]
                if self.isBalanced(sub):
                    i = j
                    break
                j += 1
        return count

    def isBalanced(self, s: str) -> bool:
        if not s:
            return True
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        if not counts:
            return True

        first_count = list(counts.values())[0]
        for value in counts.values():
            if value != first_count:
                return False
        return True