import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(X: str, Y: str) -> str:
            if Y in X:
                return X
            if X in Y:
                return Y
            for k in range(min(len(X), len(Y)), 0, -1):
                if X[-k:] == Y[:k]:
                    return X + Y[k:]
            return X + Y
        
        # Generate all permutations of a, b, c
        perms = list(itertools.permutations([a, b, c]))
        candidates = []
        
        for perm in perms:
            X, Y, Z = perm
            res1 = merge(X, Y)
            res2 = merge(res1, Z)
            candidates.append(res2)
        
        # Find the minimum length and lexicographically smallest string
        return min(candidates, key=lambda s: (len(s), s))