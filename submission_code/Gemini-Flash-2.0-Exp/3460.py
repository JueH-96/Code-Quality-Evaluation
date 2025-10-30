class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        def count_inversions(arr):
            count = 0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    if arr[i] > arr[j]:
                        count += 1
            return count

        def check_permutation(perm):
            for end, cnt in requirements:
                if count_inversions(perm[:end+1]) != cnt:
                    return False
            return True

        import itertools
        
        nums = list(range(n))
        count = 0
        for perm in itertools.permutations(nums):
            if check_permutation(list(perm)):
                count += 1
        
        return count % MOD