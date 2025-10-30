import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Function to compute the "reversed and complemented" string
    # According to problem definition: S_new[i] = complement(S_original[l+r-i])
    # For the entire string, l=0, r=N-1, so S_new[i] = complement(S_original[N-1-i])
    def compute_rev_comp(s_original):
        n = len(s_original)
        s_rev_list = [''] * n
        for i in range(n):
            original_char_at_mirror_pos = s_original[n - 1 - i]
            if original_char_at_mirror_pos == '(':
                s_rev_list[i] = ')'
            else: # original_char_at_mirror_pos == ')'
                s_rev_list[i] = '('
        return "".join(s_rev_list)

    S_rev = compute_rev_comp(S)

    # Based on sample analysis, the set of reachable strings is {S, S_rev}.
    # If S == S_rev, then only one string is reachable.
    # Otherwise, two distinct strings are reachable.
    if S == S_rev:
        sys.stdout.write("1
")
    else:
        sys.stdout.write("2
")

# Call the solve function to execute the program
solve()