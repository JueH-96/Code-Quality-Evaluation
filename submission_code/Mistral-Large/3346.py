class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        s = list(s)

        for i in range(n):
            current_char = s[i]
            distance_to_a = min(ord(current_char) - ord('a'), ord('z') - ord(current_char) + 1)

            if k >= distance_to_a:
                k -= distance_to_a
                s[i] = 'a'
            else:
                break

        if k > 0:
            i = n - 1
            while k > 0 and i >= 0:
                current_char = s[i]
                distance_to_z = min(ord('z') - ord(current_char), ord(current_char) - ord('a') + 1)

                if k >= distance_to_z:
                    k -= distance_to_z
                    s[i] = 'z'
                else:
                    s[i] = chr(ord(current_char) + k)
                    k = 0
                i -= 1

        return ''.join(s)