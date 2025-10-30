class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        original_partitions = self.get_original_partitions(s, k)
        original_count = len(original_partitions)
        max_result = original_count

        # For each original partition, try changing each character in it
        for (start, end) in original_partitions:
            for i in range(start, end + 1):
                original_char = s[i]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c == original_char:
                        continue
                    # Create modified string
                    modified = s[:i] + c + s[i+1:]
                    # Compute partitions for modified
                    current = self.count_partitions(modified, k)
                    if current > max_result:
                        max_result = current
        return max_result

    def get_original_partitions(self, s, k):
        partitions = []
        pos = 0
        n = len(s)
        while pos < n:
            freq = [0] * 26
            distinct = 0
            max_len = pos
            for i in range(pos, n):
                c = ord(s[i]) - ord('a')
                if freq[c] == 0:
                    distinct += 1
                freq[c] += 1
                if distinct > k:
                    break
                max_len = i
            partitions.append((pos, max_len))
            pos = max_len + 1
        return partitions

    def count_partitions(self, s, k):
        n = len(s)
        parts = 0
        pos = 0
        while pos < n:
            freq = [0] * 26
            distinct = 0
            max_len = pos
            for i in range(pos, n):
                c = ord(s[i]) - ord('a')
                if freq[c] == 0:
                    distinct += 1
                freq[c] += 1
                if distinct > k:
                    break
                max_len = i
            parts += 1
            pos = max_len + 1
        return parts