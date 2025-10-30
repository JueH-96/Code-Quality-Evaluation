import sys

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Use fast I/O
    readline = sys.stdin.readline
    
    # Read N and the received string T'
    try:
        n_str, t_prime = readline().split()
    except ValueError:
        # Handles empty input line at the end of file
        print(0)
        return
        
    n = int(n_str)
    len_t_prime = len(t_prime)

    ans_indices = []

    def check_one_diff(shorter: str, longer: str) -> bool:
        """
        Checks if `longer` can be formed by inserting one character into `shorter`.
        This is done by checking if the common prefix and suffix of the two strings
        collectively cover the entire `shorter` string.
        Assumes len(longer) == len(shorter) + 1.
        """
        len_shorter = len(shorter)
        len_longer = len(longer)
        
        # Find the length of the common prefix
        p_len = 0
        while p_len < len_shorter and shorter[p_len] == longer[p_len]:
            p_len += 1
        
        # Find the length of the common suffix
        s_len = 0
        while s_len < len_shorter and shorter[len_shorter - 1 - s_len] == longer[len_longer - 1 - s_len]:
            s_len += 1
        
        # The prefix and suffix must meet or overlap.
        # This condition is equivalent to: p_len >= len_shorter - s_len
        return p_len + s_len >= len_shorter

    # Process each candidate string S_i
    for i in range(1, n + 1):
        s = readline().strip()
        len_s = len(s)
        
        is_candidate = False

        # Case 1: T could be equal to T', or T could be T' with one char changed.
        # This implies len(T) == len(T').
        if len_s == len_t_prime:
            if s == t_prime:
                is_candidate = True
            else:
                diff_count = 0
                for j in range(len_s):
                    if s[j] != t_prime[j]:
                        diff_count += 1
                if diff_count == 1:
                    is_candidate = True
        
        # Case 2: T' could be T with one insertion.
        # This implies len(T') == len(T) + 1.
        elif len_s == len_t_prime - 1:
            if check_one_diff(s, t_prime):
                is_candidate = True

        # Case 3: T' could be T with one deletion.
        # This implies len(T') == len(T) - 1.
        elif len_s == len_t_prime + 1:
            if check_one_diff(t_prime, s):
                is_candidate = True
        
        # If |len(S) - len(T')| > 1, none of the conditions can be met.

        if is_candidate:
            ans_indices.append(i)

    # Print the result in the specified format
    print(len(ans_indices))
    if ans_indices:
        print(*ans_indices)

if __name__ == "__main__":
    main()