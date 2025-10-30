class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        # To store the lexicographically largest string
        largest_string = ""
        
        # Function to generate all possible splits of the word into numFriends parts
        def generate_splits(index, parts, current_split):
            nonlocal largest_string
            if index == len(word):
                if len(parts) == numFriends:
                    # Compare each part with the largest_string found so far
                    for part in parts:
                        if part > largest_string:
                            largest_string = part
                return
            if len(parts) < numFriends:
                # Continue the current part
                if current_split:
                    generate_splits(index + 1, parts + [current_split + word[index]], "")
                # Start a new part
                if len(parts) < numFriends - 1:
                    generate_splits(index + 1, parts, current_split + word[index])
            else:
                # If we already have numFriends parts, just return
                return
        
        # Start generating splits from index 0, with no parts initially
        generate_splits(0, [], "")
        
        return largest_string