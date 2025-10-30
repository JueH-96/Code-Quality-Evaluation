class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        
        if s1 == s2:
            return True

        s1_list = list(s1)
        s2_list = list(s2)

        
        def apply_swaps(s):
            
            q = [list(s)]
            visited = {tuple(s)}
            
            while q:
                curr = q.pop(0)
                
                if curr == s2_list:
                    return True
                
                for i in range(2):
                    j = i + 2
                    
                    temp = list(curr)
                    temp[i], temp[j] = temp[j], temp[i]
                    
                    if tuple(temp) not in visited:
                        visited.add(tuple(temp))
                        q.append(temp)
            
            return False
        
        return apply_swaps(s1_list)