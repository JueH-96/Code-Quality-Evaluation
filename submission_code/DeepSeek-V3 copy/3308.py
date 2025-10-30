class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        from collections import defaultdict

        # Count the frequency of each character
        freq = defaultdict(int)
        for char in s:
            freq[char] += 1

        # Find the maximum frequency
        max_freq = max(freq.values())

        # Collect characters that have the maximum frequency
        max_chars = {char for char, cnt in freq.items() if cnt == max_freq}

        # Iterate through the string in reverse to find the last occurrence of each max_char
        last_occurrence = {}
        for i in range(len(s) - 1, -1, -1):
            if s[i] in max_chars and s[i] not in last_occurrence:
                last_occurrence[s[i]] = i

        # Sort the last occurrences by their index
        sorted_last = sorted(last_occurrence.items(), key=lambda x: x[1])

        # Extract the characters in the order of their last occurrence
        result = []
        for char, idx in sorted_last:
            result.append(s[idx])

        return ''.join(result)