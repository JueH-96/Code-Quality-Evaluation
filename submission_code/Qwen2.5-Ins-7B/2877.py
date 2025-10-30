class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(x, y):
            for i in range(len(y), 0, -1):
                if y.endswith(x[:i]):
                    return x[:len(x)-i] + y
            return x + y
        
        def compare(x, y):
            if len(x) < len(y) or (len(x) == len(y) and x < y):
                return x
            return y
        
        cases = [merge(merge(a, b), c), merge(merge(a, c), b), merge(merge(b, a), c), merge(merge(b, c), a), merge(merge(c, a), b), merge(merge(c, b), a)]
        return min(cases, key=lambda x: (len(x), x))