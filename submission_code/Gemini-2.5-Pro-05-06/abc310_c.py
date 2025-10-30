import sys

def main():
    # Read the number of sticks
    N = int(sys.stdin.readline())
    
    # Use a set to store the canonical forms of the distinct sticks
    distinct_sticks = set()
    
    for _ in range(N):
        # Read the string representing the current stick
        s = sys.stdin.readline().strip()
        
        # The canonical representation of a stick is the lexicographically smaller
        # of the string s (read from one end) and its reverse s[::-1] (read from the other end).
        # This ensures that a stick described by s and a stick described by s_reversed
        # will both map to the same canonical representative.
        # For example:
        # If s = "abc", s_reversed = "cba". min("abc", "cba") is "abc".
        # If s = "cba", s_reversed = "abc". min("cba", "abc") is "abc".
        # If s = "aba" (palindrome), s_reversed = "aba". min("aba", "aba") is "aba".
        
        canonical_form = min(s, s[::-1])
            
        # Add the canonical form to the set.
        # If it's already present, the set remains unchanged.
        distinct_sticks.add(canonical_form)
            
    # The number of distinct sticks is the size of the set.
    print(len(distinct_sticks))

if __name__ == '__main__':
    main()