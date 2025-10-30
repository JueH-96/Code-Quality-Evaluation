def vertical_text_conversion():
    import sys
    input = sys.stdin.read
    data = input().strip().splitlines()
    
    N = int(data[0])
    strings = data[1:N + 1]
    
    # Determine the maximum length of the strings
    M = max(len(s) for s in strings)
    
    # Prepare the result list with empty strings
    result = [''] * M
    
    # Fill the result list
    for i in range(N):
        s = strings[i]
        for j in range(len(s)):
            result[M - j - 1] += s[j]
        for j in range(len(s), M):
            result[M - j - 1] += '*'
    
    # Ensure no string ends with '*'
    for i in range(M):
        if result[i] and result[i][-1] == '*':
            result[i] = result[i].rstrip('*')
    
    # Print the result
    for line in result:
        print(line)

vertical_text_conversion()