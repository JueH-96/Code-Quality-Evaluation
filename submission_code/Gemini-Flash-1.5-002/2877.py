class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        perms = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
        ans = ""
        min_len = float('inf')

        for x, y, z in perms:
            combined = x
            i = 0
            while i < len(y):
                if y[i:] in combined:
                    combined += y[i:]
                    break
                else:
                    j = 0
                    while j < len(combined):
                        if combined[j:] == y[:len(combined)-j]:
                            combined += y[len(combined)-j:]
                            break
                        j += 1
                    i += 1

            i = 0
            while i < len(z):
                if z[i:] in combined:
                    combined += z[i:]
                    break
                else:
                    j = 0
                    while j < len(combined):
                        if combined[j:] == z[:len(combined)-j]:
                            combined += z[len(combined)-j:]
                            break
                        j += 1
                    i += 1

            if len(combined) < min_len:
                min_len = len(combined)
                ans = combined
            elif len(combined) == min_len:
                ans = min(ans, combined)

        return ans