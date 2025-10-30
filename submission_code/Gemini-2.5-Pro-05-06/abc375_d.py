# YOUR CODE HERE
import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)

    ans = 0
    
    # num_char_so_far[c_idx] stores the count of character c_idx encountered so far (indices < k_idx)
    # sum_indices_char_so_far[c_idx] stores the sum of indices p < k_idx where S[p] has character index c_idx
    # Character index c_idx is ord(char) - ord('A')
    
    num_char_so_far = [0] * 26  # For 'A' through 'Z'
    sum_indices_char_so_far = [0] * 26

    for k_idx in range(N): # k_idx is the 0-based index for S_k
        char_code = ord(S[k_idx]) - ord('A')
        
        # For S[k_idx], we want to sum (k_idx - i_idx - 1) for all i_idx < k_idx where S[i_idx] == S[k_idx].
        # This sum is equivalent to:
        # (count of S[k_idx] before k_idx) * (k_idx - 1) - (sum of indices of S[k_idx] before k_idx)
        
        count_val = num_char_so_far[char_code]
        sum_indices_val = sum_indices_char_so_far[char_code]
        
        contribution = count_val * (k_idx - 1) - sum_indices_val
        ans += contribution
        
        # Update the stored information for the character S[k_idx]
        num_char_so_far[char_code] += 1
        sum_indices_char_so_far[char_code] += k_idx
        
    sys.stdout.write(str(ans) + "
")

if __name__ == '__main__':
    solve()
# YOUR CODE HERE