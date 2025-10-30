class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        char_count = {}
        last_positions = {}
        max_count = 0

        # Count occurrences and track last positions
        for i, char in enumerate(s):
            char_count[char] = char_count.get(char, 0) + 1
            last_positions[char] = i
            max_count = max(max_count, char_count[char])

        # Find characters with maximum count
        result = []
        for char, count in char_count.items():
            if count == max_count:
                result.append((last_positions[char], char))

        # Sort by position and extract characters
        return ''.join(char for _, char in sorted(result))