class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        n = len(words)
        count = 0
        
        def is_palindrome(s):
            return s == s[::-1]
        
        for word in words:
            if is_palindrome(word):
                count += 1
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    for x in range(len(words[i])):
                        for y in range(x+1, len(words[i])):
                            new_word = list(words[i])
                            new_word[x], new_word[y] = new_word[y], new_word[x]
                            if is_palindrome("".join(new_word)):
                                count += 1
                else:
                    for x in range(len(words[i])):
                        for y in range(len(words[j])):
                            new_word_i = list(words[i])
                            new_word_j = list(words[j])
                            new_word_i[x], new_word_j[y] = new_word_j[y], new_word_i[x]
                            if is_palindrome("".join(new_word_i)) and is_palindrome("".join(new_word_j)):
                                count += 2
        
        return count