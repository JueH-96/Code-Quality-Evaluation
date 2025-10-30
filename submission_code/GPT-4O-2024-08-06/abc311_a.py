# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    # Initialize a set to keep track of unique characters encountered
    seen_characters = set()
    
    # Iterate over the string S
    for i in range(N):
        # Add the current character to the set
        seen_characters.add(S[i])
        
        # Check if all 'A', 'B', 'C' are in the set
        if 'A' in seen_characters and 'B' in seen_characters and 'C' in seen_characters:
            # If yes, print the number of characters checked (1-based index)
            print(i + 1)
            return

# Call the main function
main()