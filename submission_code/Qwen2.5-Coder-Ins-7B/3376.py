class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        from collections import defaultdict
        
        # Create a dictionary to store the longest common suffix and its index
        suffix_dict = defaultdict(lambda: (float('inf'), float('inf')))
        
        # Iterate over the wordsContainer to populate the dictionary
        for i, word in enumerate(wordsContainer):
            for j in range(len(word)):
                suffix = word[j:]
                length = len(suffix)
                if length < suffix_dict[suffix][0] or (length == suffix_dict[suffix][0] and len(word) < suffix_dict[suffix][1]):
                    suffix_dict[suffix] = (length, len(word), i)
        
        # Create a list to store the result
        result = []
        
        # Iterate over the wordsQuery to find the answer for each query
        for query in wordsQuery:
            if query in suffix_dict:
                result.append(suffix_dict[query][2])
            else:
                result.append(suffix_dict[''][2])
        
        return result