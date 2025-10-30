class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(x: str, y: str) -> str:
            if y in x:
                return x
            for i in range(len(x), -1, -1):
                if x.endswith(y[:i]):
                    return x + y[i:]
            return x + y

        def min_str(a: str, b: str, c: str) -> str:
            return min(merge(merge(a, b), c), merge(merge(a, c), b), merge(merge(b, a), c), merge(merge(b, c), a), merge(merge(c, a), b), merge(merge(c, b), a))

        return min_str(a, b, c)