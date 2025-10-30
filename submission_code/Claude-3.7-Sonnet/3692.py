class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Split the pattern into three parts: p1, p2, p3
        parts = p.split('*')
        p1, p2, p3 = parts[0], parts[1], parts[2]
        
        # Special case: if all parts are empty, the empty substring matches
        if p1 == "" and p2 == "" and p3 == "":
            return 0
        
        min_length = float('inf')
        
        # For each valid starting position of p1
        for i in range(len(s) + 1 if p1 == "" else len(s) - len(p1) + 1):
            if p1 == "" or s[i:i+len(p1)] == p1:
                p1_end = i + len(p1)
                
                # For each valid position of p2 starting after p1
                for j in range(p1_end, len(s) + 1 if p2 == "" else len(s) - len(p2) + 1):
                    if p2 == "" or s[j:j+len(p2)] == p2:
                        p2_end = j + len(p2)
                        
                        # For each valid position of p3 starting after p2
                        for k in range(p2_end, len(s) + 1 if p3 == "" else len(s) - len(p3) + 1):
                            if p3 == "" or s[k:k+len(p3)] == p3:
                                p3_end = k + len(p3)
                                substring_length = p3_end - i
                                min_length = min(min_length, substring_length)
        
        return min_length if min_length != float('inf') else -1