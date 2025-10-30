# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Join each character of the string with a space
    result = ' '.join(S)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()