class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def check(x, y, z):
            if x.startswith(y) and y.startswith(z):
                return True
            return False

        def find_min(a, b, c):
            if len(a) <= len(b) and len(a) <= len(c):
                return a
            elif len(b) <= len(a) and len(b) <= len(c):
                return b
            else:
                return c

        def find_max(a, b, c):
            if len(a) >= len(b) and len(a) >= len(c):
                return a
            elif len(b) >= len(a) and len(b) >= len(c):
                return b
            else:
                return c

        if check(a, b, c):
            return find_min(a, b, c)
        elif check(b, a, c):
            return find_min(b, a, c)
        elif check(c, a, b):
            return find_min(c, a, b)
        else:
            return find_max(a, b, c)