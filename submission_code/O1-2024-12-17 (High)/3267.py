class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        # Check lengths in descending order
        for length in range(n, 0, -1):
            # For each character 'a' to 'z'
            for c in "abcdefghijklmnopqrstuvwxyz":
                target = c * length
                count = 0
                # Count how many times target appears (with overlap)
                for i in range(n - length + 1):
                    if s[i:i+length] == target:
                        count += 1
                # If appears at least thrice, return this length
                if count >= 3:
                    return length
        return -1