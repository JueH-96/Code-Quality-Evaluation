class Solution:
    def calculateScore(self, s: str) -> int:
        def mirror(char):
            return chr(ord('a') + ord('z') - ord(char))

        score = 0
        marked = set()
        last_seen = {}

        for i, char in enumerate(s):
            mirrored = mirror(char)
            if mirrored in last_seen:
                j = last_seen[mirrored]
                while j in marked and j < i:
                    j = last_seen[mirrored]
                if j not in marked and j < i:
                    score += i - j
                    marked.add(i)
                    marked.add(j)
            last_seen[char] = i

        return score