class Solution:
    def isSubsequence(self, source: str, pattern: str) -> bool:
        pattern_ptr = 0
        source_ptr = 0
        while pattern_ptr < len(pattern) and source_ptr < len(source):
            if pattern[pattern_ptr] == source[source_ptr]:
                pattern_ptr += 1
            source_ptr += 1
        return pattern_ptr == len(pattern)
    
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        removed_indices = []
        removed_count = 0
        for index_to_try in targetIndices:
            test_removed_indices = removed_indices + [index_to_try]
            temp_source_list = []
            indices_to_remove_set = set(test_removed_indices)
            for i in range(len(source)):
                if i not in indices_to_remove_set:
                    temp_source_list.append(source[i])
            temp_source = "".join(temp_source_list)
            if self.isSubsequence(temp_source, pattern):
                removed_indices = test_removed_indices
                removed_count += 1
            else:
                break
        return removed_count