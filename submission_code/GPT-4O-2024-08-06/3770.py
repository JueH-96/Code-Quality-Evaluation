class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        
        # The length of the resulting word should be n + m - 1
        word_length = n + m - 1
        word = ['a'] * word_length  # Start with the smallest possible letters
        
        # Iterate over str1 and try to construct the word
        for i in range(n):
            if str1[i] == 'T':
                # Check if we can place str2 starting at index i
                if i + m - 1 >= word_length:
                    # If placing str2 exceeds the length of the word, it's impossible
                    return ""
                # Place str2 at position i
                for j in range(m):
                    if word[i + j] != 'a' and word[i + j] != str2[j]:
                        # If there's a conflict, return ""
                        return ""
                    word[i + j] = str2[j]
        
        # Now, ensure that all 'F' positions do not match str2
        for i in range(n):
            if str1[i] == 'F':
                # Check if str2 is placed starting at index i
                if i + m - 1 < word_length:
                    # Check if the substring matches str2
                    if ''.join(word[i:i + m]) == str2:
                        # If it matches, we need to change something
                        # Try to change the first character of this substring to make it different
                        for j in range(m):
                            if word[i + j] != 'z':
                                word[i + j] = chr(ord(word[i + j]) + 1)
                                break
                        else:
                            # If we can't change any character, return ""
                            return ""
        
        return ''.join(word)