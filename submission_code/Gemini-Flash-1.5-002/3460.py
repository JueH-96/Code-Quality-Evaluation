class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7

        def count_inversions(perm):
            count = 0
            for i in range(len(perm)):
                for j in range(i + 1, len(perm)):
                    if perm[i] > perm[j]:
                        count += 1
            return count

        ans = 0
        import itertools
        for perm_tuple in itertools.permutations(range(n)):
            perm = list(perm_tuple)
            valid = True
            for end, cnt in requirements:
                prefix = perm[:end + 1]
                inversions = count_inversions(prefix)
                if inversions != cnt:
                    valid = False
                    break
            if valid:
                ans += 1

        return ans