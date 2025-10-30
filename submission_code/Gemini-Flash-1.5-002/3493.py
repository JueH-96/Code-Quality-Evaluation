class Solution:
    def maxOperations(self, s: str) -> int:
        count = 0
        n = len(s)
        ones_moved = 0
        
        s_list = list(s)
        
        i = 0
        while i < len(s_list) -1:
            if s_list[i] == '1' and s_list[i+1] == '0':
                j = i + 1
                while j < len(s_list) and s_list[j] == '0':
                    j += 1
                
                if j < len(s_list):
                    
                    s_list.pop(i)
                    s_list.insert(j, '1')
                    
                    
                    count +=1
                    
                    
                else:
                    s_list.pop(i)
                    s_list.append('1')
                    count += 1
                    
            else:
                i += 1

        return count