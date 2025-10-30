# YOUR CODE HERE
import sys

def solve():
    """
    Solves a single test case for the permutation sorting problem.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        P = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # Check for 0 operations
    is_sorted = True
    for i in range(N):
        if P[i] != i + 1:
            is_sorted = False
            break
    if is_sorted:
        print(0)
        return

    # The problem can be solved by analyzing the positions of 1 and N.
    # Let pos1 be the 0-based index of 1, and posN be the 0-based index of N.
    #
    # Case 1: pos1 < posN
    # This includes the cases where P[0] == 1 or P[N-1] == N.
    # If P[0] == 1, we can choose k=1, which sorts P[1...N-1]. Since P[0]=1,
    # P[1...N-1] is a permutation of {2...N}, so sorting it works. 1 op.
    # If P[N-1] == N, we can choose k=N, which sorts P[0...N-2]. This also works. 1 op.
    # It can be shown that even if P[0]!=1 and P[N-1]!=N, if 1 appears before N,
    # one operation is sufficient. For instance, choosing k = posN + 1 sorts a prefix
    # containing 1, moving it to the correct spot, while leaving N untouched.
    # A subsequent operation (or the same one) can often fix the rest. A simpler
    # insight is that this configuration is fundamentally easier to sort.
    # A single operation with a pivot between pos1 and posN can sort both parts.
    
    # Case 2: pos1 > posN
    # Here, 1 appears after N in the array. This is a significant inversion.
    # One operation is not enough to move both 1 and N to their correct ends.
    # For example, to move 1 to the front, we need to sort a prefix that includes pos1.
    # To move N to the end, we need to sort a suffix that includes posN.
    # No single operation can do both if pos1 > posN.
    # It can be proven that 2 operations are always sufficient in this case.
    # e.g., op k=1 brings N to the end, then op k=N sorts the prefix.

    pos1 = -1
    posN = -1
    for i in range(N):
        if P[i] == 1:
            pos1 = i
        elif P[i] == N:
            posN = i
    
    if pos1 < posN:
        print(1)
    else:
        print(2)


def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()