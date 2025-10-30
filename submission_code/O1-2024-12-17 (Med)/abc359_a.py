# YOUR CODE HERE
def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # The next N lines each contains a string which is either "Takahashi" or "Aoki".
    strings = data[1:]
    
    # Count how many of these strings are "Takahashi".
    count_takahashi = strings.count("Takahashi")
    
    # Output the result.
    print(count_takahashi)

# Call the main function
main()