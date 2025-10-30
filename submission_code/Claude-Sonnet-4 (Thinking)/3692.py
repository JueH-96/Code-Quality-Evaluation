class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Parse the pattern
        star_positions = [i for i, c in enumerate(p) if c == '*']
        prefix = p[:star_positions[0]]
        middle = p[star_positions[0]+1:star_positions[1]]
        suffix = p[star_positions[1]+1:]
        
        # Special case: if all parts are empty, the empty substring matches
        if prefix == "" and middle == "" and suffix == "":
            return 0
        
        n = len(s)
        min_length = float('inf')
        
        # Find all possible starting positions
        for i in range(n - len(prefix) - len(middle) - len(suffix) + 1):
            if s[i:i+len(prefix)] == prefix:
                # Find all possible ending positions
                for j in range(i + len(prefix) + len(middle) + len(suffix) - 1, n):
                    if s[j-len(suffix)+1:j+1] == suffix:
                        # Check if middle can be found in between
                        start_search = i + len(prefix)
                        end_search = j - len(suffix) + 1
                        
                        if middle == "" or s[start_search:end_search].find(middle) != -1:
                            min_length = min(min_length, j - i + 1)
                            break  # We found the shortest ending for this starting position
        
        return min_length if min_length != float('inf') else -1