# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    S = data[0]
    Q = int(data[1])
    K = list(map(int, data[2:2+Q]))
    
    # Create the transformed string T
    T = ''.join([c.lower() if c.isupper() else c.upper() for c in S])
    
    # The length of the original string S
    len_S = len(S)
    
    # The length of the new string after one operation
    len_new_S = 2 * len_S
    
    # Prepare the result list
    result = []
    
    for k in K:
        # Since the pattern repeats every len_new_S characters
        index = (k - 1) % len_new_S
        
        if index < len_S:
            result.append(S[index])
        else:
            result.append(T[index - len_S])
    
    # Print the results as a single line of space-separated characters
    print(' '.join(result))