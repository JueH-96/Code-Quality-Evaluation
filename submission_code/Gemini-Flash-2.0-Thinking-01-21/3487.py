class Solution:
    def isSubsequence(self, source: str, pattern: str, removed_indices_set: set[int]) -> bool:
        pattern_index = 0
        source_index = 0
        source_len = len(source)
        pattern_len = len(pattern)
        while source_index < source_len and pattern_index < pattern_len:
            if source_index not in removed_indices_set and source[source_index] == pattern[pattern_index]:
                pattern_index += 1
            source_index += 1
        return pattern_index == pattern_len
        
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        removed_indices_set = set()
        count = 0
        for idx_to_remove_from_target_indices in targetIndices:
            temp_removed_indices_set = set(removed_indices_set)
            temp_removed_indices_set.add(idx_to_remove_from_target_indices)
            if self.isSubsequence(source, pattern, temp_removed_indices_set):
                removed_indices_set = temp_removed_indices_set
                count += 1
            else:
                continue
        return count