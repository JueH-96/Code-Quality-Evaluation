class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        def can_construct(word, t):
            t_it = iter(t)
            return all(c in t_it for c in word)

        words.sort(key=len, reverse=True)
        count = 0
        for word in words:
            if can_construct(word, target):
                count += 1
                target = target[len(word):]
        return count