class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        count = 0
        s_list = list(s)
        
        while True:
            found = False
            for i in range(n - 1):
                if s_list[i] == '1' and s_list[i+1] == '0':
                    found = True
                    j = i + 1
                    while j < n and s_list[j] == '0':
                        j += 1
                    
                    temp = s_list[i]
                    s_list[i] = '0'
                    
                    s_list.insert(j, temp)
                    s_list.pop(i)
                    
                    count += 1
                    break
            if not found:
                break
        
        return count