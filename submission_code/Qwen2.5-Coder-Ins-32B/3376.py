from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Create a list of tuples (reversed_word, index, length) for wordsContainer
        reversed_words = [(word[::-1], i, len(word)) for i, word in enumerate(wordsContainer)]
        
        # Sort the list by length (ascending) and then by reversed_word (lexicographically)
        reversed_words.sort(key=lambda x: (x[2], x[0]))
        
        # Initialize the result list
        result = []
        
        # Process each query
        for query in wordsQuery:
            reversed_query = query[::-1]
            best_match_index = 0
            best_match_length = 0
            
            # Binary search to find the longest common suffix
            left, right = 0, len(reversed_words) - 1
            while left <= right:
                mid = (left + right) // 2
                reversed_word, index, length = reversed_words[mid]
                
                # Check if the current word has a common suffix with the query
                common_length = 0
                for i in range(min(len(reversed_query), len(reversed_word))):
                    if reversed_query[i] == reversed_word[i]:
                        common_length += 1
                    else:
                        break
                
                # Update the best match if the current one is better
                if common_length > best_match_length:
                    best_match_length = common_length
                    best_match_index = index
                
                # Move the search space
                if common_length == len(reversed_query):
                    right = mid - 1
                else:
                    left = mid + 1
            
            result.append(best_match_index)
        
        return result