class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        removed_indices = set()
        max_operations = 0

        for idx in targetIndices:
            # Add the current index to the set of removed indices
            removed_indices.add(idx)

            # Simulate the subsequence check
            p_index = 0  # Pointer for pattern
            for i in range(n):
                if i in removed_indices:
                    continue  # Skip removed characters
                if source[i] == pattern[p_index]:
                    p_index += 1
                    if p_index == len(pattern):
                        break  # All characters in pattern are matched

            # If the pattern is not fully matched, we cannot remove this index
            if p_index < len(pattern):
                removed_indices.remove(idx)
            else:
                max_operations += 1  # We successfully removed this character

        return max_operations