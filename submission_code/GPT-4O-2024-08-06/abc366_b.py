# YOUR CODE HERE
def vertical_text_conversion():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    strings = data[1:N+1]
    
    # Determine the maximum length of the strings
    M = max(len(s) for s in strings)
    
    # Initialize the result list with empty strings
    result = ['' for _ in range(M)]
    
    # Fill the result list with characters from the input strings
    for i in range(M):
        for j in range(N):
            if i < len(strings[j]):
                result[i] = strings[j][i] + result[i]
            else:
                result[i] = '*' + result[i]
    
    # Remove trailing '*' from each string in the result
    for i in range(M):
        result[i] = result[i].rstrip('*')
    
    # Print the result
    for line in result:
        print(line)

vertical_text_conversion()