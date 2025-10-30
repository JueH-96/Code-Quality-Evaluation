# YOUR CODE HERE
def main():
    # Read the input string S
    S = input()
    
    # The string S is generated as follows:
    # Start with S = "|"
    # For each A_i in the sequence A:
    #   Append A_i copies of "-" to S
    #   Append one "|" to S
    # This means S will have the form: |<A_1 hyphens>|<A_2 hyphens>|...|<A_N hyphens>|
    
    # We can split the string S by the '|' character.
    # For example, if S = "|---|-|----|", then S.split('|') yields ['', '---', '-', '----', ''].
    parts = S.split('|')
    
    # The first part (parts[0]) is the empty string resulting from the initial '|'.
    # The last part (parts[len(parts)-1] or parts[-1]) is the empty string
    # resulting from the final '|'.
    # The segments of hyphens corresponding to A_1, A_2, ..., A_N are
    # located at indices 1, 2, ..., len(parts)-2 of the 'parts' list.
    # In Python slicing, this sublist is parts[1:-1].
    hyphen_blocks = parts[1:-1]
    
    # The sequence A consists of the lengths of these hyphen blocks.
    # Since A_i are positive integers (A_i >= 1), each hyphen_block will be non-empty
    # and its length will be A_i.
    a_values = []
    for block in hyphen_blocks:
        a_values.append(len(block))
        
    # Print the reconstructed sequence A with elements separated by spaces.
    # The `*` operator unpacks the list `a_values` into individual arguments
    # for the print function. By default, print separates arguments with a space
    # and adds a newline at the end.
    print(*a_values)

if __name__ == '__main__':
    main()