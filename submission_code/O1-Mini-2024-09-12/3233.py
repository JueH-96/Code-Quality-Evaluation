class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        from collections import defaultdict

        n = len(s)
        max_partitions = 0

        # Function to compute partition count without any change
        def compute_partitions(s):
            count = 0
            freq = defaultdict(int)
            distinct = 0
            left = 0
            for right in range(len(s)):
                if freq[s[right]] == 0:
                    distinct += 1
                freq[s[right]] += 1
                while distinct > k:
                    freq[s[left]] -= 1
                    if freq[s[left]] == 0:
                        distinct -= 1
                    left += 1
                # When we reach a point where adding next character would exceed
                # We count the current window as a partition
                # but since we are taking maximum partitions, we continue
            # To count partitions, iterate again similar to sliding window
            count = 0
            freq = defaultdict(int)
            distinct = 0
            left = 0
            for right in range(len(s)):
                if freq[s[right]] == 0:
                    distinct += 1
                freq[s[right]] += 1
                if distinct > k:
                    count += 1
                    # Reset for next partition
                    freq = defaultdict(int)
                    freq[s[right]] = 1
                    distinct = 1
                    left = right
            if left < len(s):
                count +=1
            return count

        # Initial partition count without any change
        original_partitions = compute_partitions(s)
        max_partitions = original_partitions

        # Try changing each character to every other possible character
        for i in range(n):
            original_char = s[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == original_char:
                    continue
                new_s = s[:i] + c + s[i+1:]
                partitions = compute_partitions(new_s)
                if partitions > max_partitions:
                    max_partitions = partitions
        return max_partitions