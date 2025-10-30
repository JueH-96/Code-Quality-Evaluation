class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(x, y):
            if y in x:
                return x
            for i in range(len(y), -1, -1):
                if x.endswith(y[:i]):
                    return x + y[i:]
            return x + y
        
        def combine(x, y, z):
            return min(merge(x, merge(y, z)), merge(x, merge(z, y)), merge(y, merge(x, z)), merge(y, merge(z, x)), merge(z, merge(x, y)), merge(z, merge(y, x)))
        
        return combine(a, b, c)