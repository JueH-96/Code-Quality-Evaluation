import sys

def solve():
    # Read N, the length of the string S.
    # While N is given, len(S) can also be used if S is read first.
    N = int(sys.stdin.readline())
    
    # Read the string S and remove any trailing newline.
    S = sys.stdin.readline().strip()
    
    # Read Q, the number of operations.
    Q = int(sys.stdin.readline())

    # Initialize the transformation map.
    # 'mapping' is a list of 26 characters.
    # mapping[i] will store what the character chr(ord('a') + i)
    # (i.e., the i-th lowercase letter 'a', 'b', ...) will eventually become.
    # Initially, each character maps to itself.
    mapping = [chr(ord('a') + i) for i in range(26)]

    # Process each of the Q operations.
    for _ in range(Q):
        # Read the two characters for the current operation (c_i, d_i).
        # .split() separates them by whitespace.
        c_i, d_i = sys.stdin.readline().split()

        # Optimization: If the source character c_i is the same as the destination d_i,
        # this operation makes no change. We can skip iterating over the mapping.
        if c_i == d_i:
            continue
        
        # Apply the transformation rule:
        # Iterate through all 26 possible lowercase English letters.
        # For each letter (represented by its index 'j' from 0 to 25),
        # if its current mapped value (mapping[j]) is 'c_i',
        # then update mapping[j] to 'd_i'.
        # This correctly propagates transformations (e.g., if 'a' becomes 'x',
        # and then 'x' becomes 'y', 'a' will ultimately become 'y').
        for j in range(26):
            if mapping[j] == c_i:
                mapping[j] = d_i
    
    # After all Q operations, the 'mapping' list contains the final form
    # for each original lowercase character.

    # Now, construct the final string by applying this mapping to the original string S.
    # Since strings in Python are immutable, it's efficient to build a list of characters
    # and then join them into a single string at the end.
    result_chars = []
    for char_s in S:
        # Calculate the index corresponding to the current character from S.
        # 'a' corresponds to index 0, 'b' to 1, and so on.
        idx = ord(char_s) - ord('a')
        
        # Look up the transformed character using the calculated index
        # and append it to our list of result characters.
        result_chars.append(mapping[idx])
    
    # Join the list of characters into a single string and print it to standard output.
    print("".join(result_chars))

# This ensures that the solve() function is called when the script is executed.
if __name__ == '__main__':
    solve()