class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        
        # Generate all possible splits
        def generate_splits(start, parts):
            if parts == 1:
                yield [word[start:]]
            else:
                for i in range(start + 1, n - parts + 2):
                    for rest in generate_splits(i, parts - 1):
                        yield [word[start:i]] + rest
        
        # Set to store all unique strings from all splits
        all_strings = set()
        
        # Generate all possible splits and add strings to the set
        for split in generate_splits(0, numFriends):
            all_strings.update(split)
        
        # Return the lexicographically largest string
        return max(all_strings)