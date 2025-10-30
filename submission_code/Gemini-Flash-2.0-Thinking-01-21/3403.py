class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(sub):
            if not sub:
                return True
            counts = {}
            for char in sub:
                counts[char] = counts.get(char, 0) + 1
            if not counts:
                return True
            first_count = None
            for count in counts.values():
                if first_count is None:
                    first_count = count
                if count != first_count:
                    return False
            return True

        n = len(s)
        if n == 0:
            return 0
        
        count = 0
        start_index = 0
        while start_index < n:
            count += 1
            best_end_index = start_index
            for end_index in range(start_index, n):
                substring = s[start_index:end_index+1]
                if is_balanced(substring):
                    best_end_index = end_index
                else:
                    break # optimization: no need to check longer substrings if current one is not balanced
            start_index = best_end_index + 1
        return count