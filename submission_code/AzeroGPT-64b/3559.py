from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_valid = False

class Solution:
    def minValidStrings(self, words, target):
        def insertIntoTrie(word):
            current = prefixTrie
            for char in word:
                current = current.children[char]
            current.is_valid = True
        
        def findMinValidStrings(s):
            current = prefixTrie
            validStringsCount = 0
            for char in s:
                if current.is_valid:
                    validStringsCount += 1
                    current = prefixTrie.children[char]
                else:
                    current = current.children[char]
            return validStringsCount + (current.is_valid)
        
        prefixTrie = TrieNode()
        for word in words:
            insertIntoTrie(word)
        
        for i in range(len(target)-1, -1, -1):
            if findMinValidStrings(target[i:]) == -1:
                return -1
        return findMinValidStrings(target)