class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def check(sub, s):
            return sub in s

        permutations = [a + b + c, a + c + b, b + a + c, b + c + a, c + a + b, c + b + a]
        for i in range(len(permutations)):
            permutations[i] = list(permutations[i])
            for j in range(len(permutations[i]) - 1, -1, -1):
                if not check(a, ''.join(permutations[i])) or not check(b, ''.join(permutations[i])) or not check(c, ''.join(permutations[i])):
                    permutations[i].pop(j)
            permutations[i] = ''.join(permutations[i])
        permutations.sort()
        return permutations[0]