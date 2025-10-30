class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        from itertools import combinations
        
        n = len(word)
        if numFriends == 1:
            return word  # If there's only one friend, the whole word is the answer
        
        # Generate all possible combinations of split points
        split_points = list(combinations(range(1, n), numFriends - 1))
        max_string = ""
        
        for points in split_points:
            # Create the splits based on the current combination of split points
            last_index = 0
            split_words = []
            for point in points:
                split_words.append(word[last_index:point])
                last_index = point
            split_words.append(word[last_index:])  # Add the last segment
            
            # Join the split words to form the combined string
            combined_string = ''.join(split_words[:-1])  # Exclude the last word for comparison
            max_string = max(max_string, combined_string)
        
        return max_string