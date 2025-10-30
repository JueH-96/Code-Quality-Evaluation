from typing import List, Set, Dict

class Solution:
    # memo_swapped_versions will store mappings from an integer to the set of
    # integers obtainable from it by at most one digit swap.
    # It's defined as an instance variable and initialized in countPairs.
    memo_swapped_versions: Dict[int, Set[int]]

    def _get_swapped_versions(self, n_val: int) -> Set[int]:
        """
        Computes and returns the set of numbers obtainable from n_val
        by performing at most one swap of digits. Results are memoized.
        """
        if n_val in self.memo_swapped_versions:
            return self.memo_swapped_versions[n_val]

        s_digits = list(str(n_val))  # Convert number to list of characters (digits)
        L = len(s_digits)
        
        # Initialize set with the number itself (0 swaps)
        versions: Set[int] = {n_val}

        # Generate numbers by 1 swap
        # Swaps are only possible if there are at least two digits.
        # The loops for i and j correctly handle L < 2 (no iterations).
        for i in range(L):
            for j in range(i + 1, L):  # j > i ensures distinct positions and each pair (i,j) is considered once
                temp_digits = list(s_digits)  # Create a mutable copy for swapping
                
                # Perform swap
                temp_digits[i], temp_digits[j] = temp_digits[j], temp_digits[i]
                
                # Form new number string and convert to int.
                # int() handles leading zeros (e.g., int("01") == 1).
                swapped_num_str = "".join(temp_digits)
                versions.add(int(swapped_num_str))
        
        self.memo_swapped_versions[n_val] = versions
        return versions

    def countPairs(self, nums: List[int]) -> int:
        # Initialize memoization cache for this specific call to countPairs.
        # This ensures freshness if the Solution object is reused by the judge.
        self.memo_swapped_versions = {}
        
        n = len(nums)
        ans = 0
        
        # Iterate through all unique pairs of indices (i, j) such that i < j
        for i in range(n):
            for j in range(i + 1, n):
                num_i = nums[i]
                num_j = nums[j]
                
                # Definition of "almost equal":
                # num_j can be formed from num_i by at most one swap (num_j is in S(num_i)), OR
                # num_i can be formed from num_j by at most one swap (num_i is in S(num_j)).

                # Check if num_j is in S(num_i)
                versions_from_num_i = self._get_swapped_versions(num_i)
                if num_j in versions_from_num_i:
                    ans += 1
                    continue  # Pair counted, move to the next pair
                
                # If not, check if num_i is in S(num_j)
                # This check is needed because S(x) containing y doesn't imply S(y) contains x
                # (e.g., x=10, y=1: S(10)={10,1} contains 1, but S(1)={1} does not contain 10).
                versions_from_num_j = self._get_swapped_versions(num_j)
                if num_i in versions_from_num_j:
                    ans += 1
        
        return ans