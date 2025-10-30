class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_two(s1, s2):
            # If s2 is already a substring of s1, return s1
            if s2 in s1:
                return s1
            # If s1 is already a substring of s2, return s2
            if s1 in s2:
                return s2
            
            # Try to find maximum overlap where suffix of s1 matches prefix of s2
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            
            # Return merged string
            return s1 + s2[max_overlap:]
        
        def merge_three_ordered(s1, s2, s3):
            # First merge s1 and s2
            temp = merge_two(s1, s2)
            # Then merge the result with s3
            return merge_two(temp, s3)
        
        strings = [a, b, c]
        permutations = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ]
        
        results = []
        for perm in permutations:
            s1, s2, s3 = strings[perm[0]], strings[perm[1]], strings[perm[2]]
            result = merge_three_ordered(s1, s2, s3)
            results.append(result)
        
        # Find the shortest result, and if there are ties, the lexicographically smallest
        min_len = min(len(r) for r in results)
        candidates = [r for r in results if len(r) == min_len]
        return min(candidates)