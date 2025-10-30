def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Expected characters at odd and even indices
    expected_odd = 'i'
    expected_even = 'o'
    
    # Count the number of insertions needed
    insertions = 0
    
    # Iterate over the string and check each character
    for index, char in enumerate(S):
        if (index % 2 == 0 and char != expected_odd) or (index % 2 == 1 and char != expected_even):
            # If the character is not as expected, we need to insert the correct character before it
            insertions += 1
            # Also, we need to adjust the index to simulate that we inserted a character
            # This means the current character will now be considered at the next position
            # So we need to check it again in the next loop iteration with the correct expected value
            # We decrement index to keep it effectively the same in the next iteration after increment
            index -= 1
    
    # Print the result
    print(insertions)

if __name__ == "__main__":
    main()