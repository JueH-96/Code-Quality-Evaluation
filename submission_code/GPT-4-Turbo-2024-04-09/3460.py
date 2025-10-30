class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Sort requirements by end_i
        requirements.sort()
        
        # Maximum inversions possible in a sequence of length n is n*(n-1)/2
        max_inversions = n * (n - 1) // 2
        
        # dp[k][i][j] will be the number of permutations of the first k elements
        # that have exactly j inversions and satisfy all requirements up to index i in requirements
        dp = [[[0 for _ in range(max_inversions + 1)] for _ in range(len(requirements))] for _ in range(n + 1)]
        
        # Initialize dp for k = 0 (empty permutation)
        for i in range(len(requirements)):
            if requirements[i][1] == 0:
                dp[0][i][0] = 1
        
        # Fill dp table
        for k in range(1, n + 1):
            # Sum up all previous inversions possibilities for the new size k
            for i in range(len(requirements)):
                end_i, cnt_i = requirements[i]
                if k - 1 <= end_i:
                    # We can only consider this requirement if k-1 elements can reach up to end_i
                    for j in range(max_inversions + 1):
                        # Calculate the number of ways to form j inversions with k elements
                        # by adding the k-th element in each possible position
                        for pos in range(k):  # position to insert the new element
                            inversions_added = pos
                            if j >= inversions_added:
                                dp[k][i][j] += dp[k-1][i][j-inversions_added]
                                dp[k][i][j] %= MOD
        
        # We need to find the number of permutations of length n that satisfy all requirements
        result = 0
        for j in range(max_inversions + 1):
            valid = True
            for i, (end_i, cnt_i) in enumerate(requirements):
                if end_i == n - 1 and cnt_i != j:
                    valid = False
                    break
            if valid:
                result += dp[n][len(requirements) - 1][j]
                result %= MOD
        
        return result