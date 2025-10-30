class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Function to find all starting indices of a substring in s
        def find_indices(s, substr):
            n = len(s)
            m = len(substr)
            indices = []
            for i in range(n - m + 1):
                if s[i:i+m] == substr:
                    indices.append(i)
            return indices
        
        # Find all indices where a appears in s
        a_indices = find_indices(s, a)
        # Find all indices where b appears in s
        b_indices = find_indices(s, b)
        
        beautiful = []
        # For each a index, check if there is a b index within k distance
        for i in a_indices:
            # Binary search to find the closest b indices
            left = 0
            right = len(b_indices) - 1
            found = False
            while left <= right:
                mid = (left + right) // 2
                j = b_indices[mid]
                if abs(j - i) <= k:
                    found = True
                    break
                elif j < i:
                    left = mid + 1
                else:
                    right = mid - 1
            if found:
                beautiful.append(i)
        
        return beautiful