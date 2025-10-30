class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        from itertools import combinations

        # Generate all possible splits of the word into numFriends non-empty strings
        splits = []
        for indices in combinations(range(1, len(word)), numFriends - 1):
            start = 0
            split = []
            for index in indices:
                split.append(word[start:index])
                start = index
            split.append(word[start:])
            splits.append(split)

        # Find the lexicographically largest string from all splits
        max_string = ""
        for split in splits:
            for s in split:
                if s > max_string:
                    max_string = s

        return max_string