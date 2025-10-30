class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern around '*'
        parts = p.split('*')
        
        # Extract parts
        left = parts[0]
        middle = parts[1]
        right = parts[2]
        
        n = len(s)
        min_len = float('inf')
        
        # Precompute all possible starting positions of `left` in `s`
        left_positions = []
        if left:
            left_len = len(left)
            for i in range(n - left_len + 1):
                if s[i:i + left_len] == left:
                    left_positions.append(i)
        else:
            left_positions.append(-1)  # If left part is empty, it can match at position -1
        
        # Precompute all possible ending positions of `right` in `s`
        right_positions = []
        if right:
            right_len = len(right)
            for i in range(n - right_len + 1):
                if s[i:i + right_len] == right:
                    right_positions.append(i + right_len - 1)
        else:
            right_positions.append(n)  # If right part is empty, it can match at position n
        
        # Now find the shortest substring that matches the pattern
        from bisect import bisect_left
        
        for start in left_positions:
            # We need the end of the middle to be at least `start + len(left)`
            min_end_index = start + len(left) + len(middle)
            # Find the smallest end in right_positions that is >= min_end_index
            idx = bisect_left(right_positions, min_end_index)
            if idx < len(right_positions):
                end = right_positions[idx]
                if end >= min_end_index:
                    # Calculate the length of the substring
                    min_len = min(min_len, end - start + 1)
        
        return min_len if min_len != float('inf') else -1