class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        removed = [False] * n
        
        def remove_chars(s_list, removed):
            new_s = ""
            new_removed = [False] * len(s_list)
            
            removed_count = 0
            for i in range(len(s_list)):
                if removed[i]:
                    removed_count += 1
                else:
                    new_s += s_list[i]
            
            if not new_s:
                return ""
            
            
            temp_s = list(new_s)
            temp_removed = [False] * len(new_s)
            
            for char_code in range(ord('a'), ord('z') + 1):
                char = chr(char_code)
                found = False
                for i in range(len(temp_s)):
                    if not temp_removed[i] and temp_s[i] == char:
                        temp_removed[i] = True
                        found = True
                        break
            
            final_s = ""
            for i in range(len(temp_s)):
                if not temp_removed[i]:
                    final_s += temp_s[i]
                    
            return final_s

        
        prev_s = s
        while s:
            
            prev_s = s
            s = remove_chars(list(s), [False] * len(s))

        return prev_s