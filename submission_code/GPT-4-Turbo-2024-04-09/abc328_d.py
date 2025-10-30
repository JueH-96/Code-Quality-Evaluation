def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # We will use a list to simulate a mutable string for efficiency
    result = []
    
    # We will traverse the string and use the result list to build the final string
    i = 0
    while i < len(S):
        # Append the current character to the result
        result.append(S[i])
        
        # Check if the last three characters in result form "ABC"
        if len(result) >= 3 and result[-3:] == ['A', 'B', 'C']:
            # Remove the last three characters
            result.pop()
            result.pop()
            result.pop()
        
        i += 1
    
    # Join the result list to form the final string
    print(''.join(result))

if __name__ == "__main__":
    main()