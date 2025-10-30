# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Filter out all characters except '2'
    result = ''.join([char for char in S if char == '2'])
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()