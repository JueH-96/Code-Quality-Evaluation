class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        positions = []
        for i, ch in enumerate(s):
            if ch == c:
                positions.append(i)
                
        n = len(positions)
        # Each pair of positions (including the same index as start and end) forms a valid substring
        return (n * (n + 1)) // 2