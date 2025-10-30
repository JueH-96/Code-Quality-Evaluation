class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        max_partitions = 0

        def count_partitions(current_s, k):
            partitions = 0
            start = 0
            n_current = len(current_s)
            while start < n_current:
                best_end = start
                distinct_chars = set()
                for end in range(start, n_current):
                    distinct_chars.add(current_s[end])
                    if len(distinct_chars) <= k:
                        best_end = end + 1
                    else:
                        break
                if best_end == start:
                    return -1  # Should not happen
                start = best_end
                partitions += 1
            return partitions

        # Case 0: No change
        max_partitions = max(max_partitions, count_partitions(s, k))

        # Case 1: Change one character
        for i in range(n):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                new_char = chr(char_code)
                if new_char != original_char:
                    modified_s = list(s)
                    modified_s[i] = new_char
                    modified_s_str = "".join(modified_s)
                    max_partitions = max(max_partitions, count_partitions(modified_s_str, k))

        return max_partitions