from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Create a dictionary to store the suffixes and their corresponding indices
        suffix_dict = {}
        min_length_index = 0
        
        # Populate the dictionary with suffixes and their indices
        for i, word in enumerate(wordsContainer):
            for j in range(len(word)):
                suffix = word[j:]
                if suffix not in suffix_dict:
                    suffix_dict[suffix] = i
                elif len(wordsContainer[suffix_dict[suffix]]) > len(word):
                    suffix_dict[suffix] = i
            if len(word) < len(wordsContainer[min_length_index]):
                min_length_index = i
        
        # Handle the case where no common suffix is found
        suffix_dict[''] = min_length_index
        
        # Find the indices for each query
        result = []
        for query in wordsQuery:
            for j in range(len(query)):
                suffix = query[j:]
                if suffix in suffix_dict:
                    result.append(suffix_dict[suffix])
                    break
            else:
                result.append(suffix_dict[''])
        
        return result