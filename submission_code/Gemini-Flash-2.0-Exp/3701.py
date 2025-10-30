class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        if n < 3:
            return ""

        def is_good(s):
            if not s:
                return False
            count = 1
            for i in range(1, len(s)):
                if s[i] == s[i - 1]:
                    count += 1
                else:
                    if count < 3:
                        return False
                    count = 1
            if count < 3:
                return False
            return True

        def solve():
            q = [(caption, 0)]
            visited = {caption}
            
            while q:
                curr_caption, cost = q.pop(0)
                
                if is_good(curr_caption):
                    return curr_caption, cost
                
                if cost > n:
                    continue
                
                for i in range(n):
                    
                    # Option 1: Change to previous character
                    if curr_caption[i] != 'a':
                        new_caption1 = list(curr_caption)
                        new_caption1[i] = chr(ord(curr_caption[i]) - 1)
                        new_caption1 = "".join(new_caption1)
                        
                        if new_caption1 not in visited:
                            q.append((new_caption1, cost + 1))
                            visited.add(new_caption1)
                    
                    # Option 2: Change to next character
                    if curr_caption[i] != 'z':
                        new_caption2 = list(curr_caption)
                        new_caption2[i] = chr(ord(curr_caption[i]) + 1)
                        new_caption2 = "".join(new_caption2)
                        
                        if new_caption2 not in visited:
                            q.append((new_caption2, cost + 1))
                            visited.add(new_caption2)
            return "", float('inf')

        best_caption = ""
        min_cost = float('inf')
        
        q = [(caption, 0)]
        visited = {caption}
        
        while q:
            curr_caption, cost = q.pop(0)
            
            if is_good(curr_caption):
                if cost < min_cost:
                    min_cost = cost
                    best_caption = curr_caption
                elif cost == min_cost and curr_caption < best_caption:
                    best_caption = curr_caption
                
                
            if cost > n:
                continue
            
            for i in range(n):
                
                # Option 1: Change to previous character
                if curr_caption[i] != 'a':
                    new_caption1 = list(curr_caption)
                    new_caption1[i] = chr(ord(curr_caption[i]) - 1)
                    new_caption1 = "".join(new_caption1)
                    
                    if new_caption1 not in visited:
                        q.append((new_caption1, cost + 1))
                        visited.add(new_caption1)
                
                # Option 2: Change to next character
                if curr_caption[i] != 'z':
                    new_caption2 = list(curr_caption)
                    new_caption2[i] = chr(ord(curr_caption[i]) + 1)
                    new_caption2 = "".join(new_caption2)
                    
                    if new_caption2 not in visited:
                        q.append((new_caption2, cost + 1))
                        visited.add(new_caption2)
        
        return best_caption