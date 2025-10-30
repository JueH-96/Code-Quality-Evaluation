class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_dict = {}
        pair_count = 0
        for word in words:
            reversed_word = word[::-1]
            if reversed_word in word_dict:
                pair_count += 1
                del word_dict[reversed_word]
            else:
                word_dict[word] = 1
        return pair_count