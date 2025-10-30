import collections
import sys

# Global variables to store input values, state for recursion, and result.
N_val = 0
K_val = 0
s_char_counts = None  # collections.Counter for character frequencies
s_unique_chars = None # Sorted list of unique characters from S
result_count = 0
current_perm_list = [] # Stores the permutation being built

def is_palindrome_suffix(length_to_check):
    """
    Checks if the last 'length_to_check' characters of 'current_perm_list' form a palindrome.
    """
    # Determine the starting index of the suffix in current_perm_list
    start_idx = len(current_perm_list) - length_to_check
    
    # Compare characters from both ends of the suffix, moving inwards
    for j in range(length_to_check // 2):
        if current_perm_list[start_idx + j] != current_perm_list[start_idx + length_to_check - 1 - j]:
            return False # Characters do not match, so not a palindrome
    return True # All checked character pairs matched, it is a palindrome

def generate_permutations_recursive(num_placed_chars):
    """
    Recursively generates permutations and counts those valid according to the problem criteria.
    'num_placed_chars' is the count of characters currently in current_perm_list.
    """
    global result_count # Allow modification of the global counter

    # Pruning step:
    # If current_perm_list has at least K_val characters,
    # check if its suffix of length K_val is a palindrome.
    if num_placed_chars >= K_val:
        if is_palindrome_suffix(K_val):
            # If it is a palindrome, any full permutation derived from this prefix
            # will contain this palindrome. So, prune this search branch.
            return

    # Base case for recursion:
    # If N_val characters have been placed, a full permutation is formed.
    if num_placed_chars == N_val:
        # Since it passed all pruning checks up to this point,
        # this permutation is valid (does not contain any K-length palindrome).
        result_count += 1
        return

    # Recursive step:
    # Try adding each available character to extend the current_perm_list.
    for char_val in s_unique_chars: # Iterate over unique chars in a fixed order
        if s_char_counts[char_val] > 0: # If this character is still available
            # "Use" the character:
            s_char_counts[char_val] -= 1    # Decrement its available count
            current_perm_list.append(char_val) # Add to current permutation
            
            # Recursively call to place the next character
            generate_permutations_recursive(num_placed_chars + 1)
            
            # "Un-use" the character (backtrack):
            current_perm_list.pop()         # Remove from current permutation
            s_char_counts[char_val] += 1    # Restore its available count

def main():
    # Make global variables accessible for modification within main
    global N_val, K_val, s_char_counts, s_unique_chars, current_perm_list, result_count
    
    # Read N and K from the first line of input
    line1 = sys.stdin.readline().split()
    N_val = int(line1[0])
    K_val = int(line1[1])
    
    # Read the string S from the second line of input
    S_str = sys.stdin.readline().strip()
    
    # Initialize character counts and the sorted list of unique characters
    s_char_counts = collections.Counter(S_str)
    s_unique_chars = sorted(s_char_counts.keys()) # Sorting ensures consistent behavior
    
    # Reset state variables (mainly for scenarios with multiple test cases, though not here)
    current_perm_list = [] 
    result_count = 0
    
    # Start the recursive generation process. Initially, 0 characters are placed.
    generate_permutations_recursive(0)
    
    # Print the final count of valid permutations
    print(result_count)

if __name__ == '__main__':
    # Python's default recursion limit is usually 1000 or 3000.
    # N=10 is very shallow, so default limit is fine.
    # If N were larger, might need: sys.setrecursionlimit(desired_limit)
    main()