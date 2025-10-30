# YOUR CODE HERE
import sys
import bisect

def main():
    """
    Main function to solve the problem.
    It reads N, M, S, and T from stdin, computes the result, and prints it to stdout.
    """
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        n, m = map(int, line1.split())
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except (IOError, ValueError):
        return

    s_list = list(s)
    
    # pos[d] will store a sorted list of indices i where s_list[i] == d.
    pos = [[] for _ in range(10)]
    for i, char_digit in enumerate(s_list):
        pos[int(char_digit)].append(i)

    # Precompute suffix maximums for T.
    # t_suffix_max[k] stores the maximum digit in T[k:].
    # We add an extra element at the end to simplify boundary checks.
    t_suffix_max = ['0'] * (m + 1)
    if m > 0:
        t_suffix_max[m - 1] = t[m - 1]
        for i in range(m - 2, -1, -1):
            t_suffix_max[i] = max(t[i], t_suffix_max[i + 1])

    # Process each character of T in the given order.
    for k in range(m):
        d_t_char = t[k]
        d_t = int(d_t_char)
        
        # A digit from T is considered "best from now" if it's strictly
        # greater than any subsequent digit in T.
        is_best_from_now = (d_t_char > t_suffix_max[k + 1])

        i_to_replace = -1
        
        if is_best_from_now:
            # If d_t is the best we'll see, use it for the maximum possible
            # lexicographical gain by improving the leftmost possible digit.
            # We look for the smallest digit v < d_t, and pick its leftmost occurrence.
            for v in range(d_t):
                if pos[v]:
                    i_to_replace = pos[v][0]
                    break
        else:
            # If a better or equal digit is coming, we don't want to use d_t on a "prime"
            # spot. We still seek an improvement, but one that is "less optimal".
            # We look for the largest digit v < d_t, and pick its rightmost occurrence.
            for v in range(d_t - 1, -1, -1):
                if pos[v]:
                    i_to_replace = pos[v][-1]
                    break
        
        if i_to_replace == -1:
            # If no improvement is possible (all digits in S are >= d_t),
            # we must still perform the replacement. To minimize the damage,
            # we change the rightmost digit of S.
            i_to_replace = n - 1
        
        # Get the old digit at the target index before overwriting.
        v_old = int(s_list[i_to_replace])
        
        # Perform the replacement.
        s_list[i_to_replace] = d_t_char
        
        # Update the 'pos' data structure to reflect the change.
        pos[v_old].remove(i_to_replace)
        bisect.insort_left(pos[d_t], i_to_replace)

    print("".join(s_list))

if __name__ == "__main__":
    main()