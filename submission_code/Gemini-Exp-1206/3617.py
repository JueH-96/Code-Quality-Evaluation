class Solution:
    def possibleStringCount(self, word: str) -> int:
        n = len(word)
        count = 0
        
        def generate_strings(index, current_string):
            nonlocal count
            if index == n:
                count += 1
                return
            
            generate_strings(index + 1, current_string + word[index])
            
            if index > 0 and word[index] == word[index-1]:
                return
            
            i = index + 1
            while i < n and word[i] == word[index]:
                generate_strings(i + 1, current_string + word[index])
                i += 1

        generate_strings(0, "")
        return count