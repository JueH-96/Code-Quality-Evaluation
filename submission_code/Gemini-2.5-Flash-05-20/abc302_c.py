import sys
import itertools

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())

    # Read N strings into a list
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Helper function to calculate the number of differing characters between two strings.
    # This function assumes s1 and s2 have the same length M.
    def calculate_diff(s1, s2):
        diff_count = 0
        for i in range(M):
            if s1[i] != s2[i]:
                diff_count += 1
        return diff_count

    # Generate all possible permutations of the input strings.
    # Since N is very small (up to 8), N! permutations is feasible.
    # N=8 implies 8! = 40,320 permutations.
    for p in itertools.permutations(S):
        # 'p' is a tuple representing one specific ordered arrangement of the strings.
        # For example, if S = ["a", "b", "c"], p could be ("b", "a", "c").
        
        is_current_permutation_valid = True
        # Iterate through adjacent pairs in the current permutation (T_i, T_{i+1}).
        # We need to check from index 0 up to N-2 (inclusive) to cover all N-1 pairs.
        for i in range(N - 1):
            # Check the difference between the current string p[i] and the next string p[i+1].
            if calculate_diff(p[i], p[i+1]) != 1:
                is_current_permutation_valid = False
                break # If any pair doesn't satisfy the condition (differs by exactly 1 character),
                      # then this entire permutation is invalid. No need to check further pairs
                      # within this permutation; move to the next one.
        
        # If the inner loop completed without setting is_current_permutation_valid to False,
        # it means this permutation satisfies all conditions (all adjacent pairs differ by 1).
        if is_current_permutation_valid:
            print("Yes")
            return # We found a valid sequence, so we can print "Yes" and exit the program immediately.

    # If the program reaches this point, it means that none of the generated permutations
    # satisfied the condition.
    print("No")

# This ensures that the solve() function is called when the script is executed.
if __name__ == '__main__':
    solve()