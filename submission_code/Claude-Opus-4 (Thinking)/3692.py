class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern by '*' to get three parts
        parts = p.split('*')
        part1, part2, part3 = parts[0], parts[1], parts[2]
        
        # If all parts are empty, return 0 (empty substring matches)
        if not part1 and not part2 and not part3:
            return 0
        
        min_length = float('inf')
        
        # Try all possible starting positions
        for start in range(len(s) + 1):
            # If part1 is not empty, it must start at 'start'
            if part1:
                if start + len(part1) > len(s) or s[start:start+len(part1)] != part1:
                    continue
                pos = start + len(part1)
            else:
                pos = start
            
            # Find the earliest occurrence of part2 starting from pos
            if part2:
                idx = s.find(part2, pos)
                if idx == -1:
                    continue
                pos = idx + len(part2)
            
            # Find the earliest occurrence of part3 starting from pos
            if part3:
                idx = s.find(part3, pos)
                if idx == -1:
                    continue
                pos = idx + len(part3)
            
            # Calculate the length of the match
            length = pos - start
            min_length = min(min_length, length)
        
        return min_length if min_length != float('inf') else -1