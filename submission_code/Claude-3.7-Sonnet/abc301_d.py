def solve(S, N):
    # Convert N to binary without '0b' prefix
    binary_N = bin(N)[2:]
    
    # Case 1: S is shorter than binary_N
    # Any number we can form from S will be less than N, so maximize it by replacing all ? with 1
    if len(S) < len(binary_N):
        return int(S.replace('?', '1'), 2)
    
    # Case 2: S is longer than binary_N
    # Need to ensure the leading digits don't make it exceed N
    if len(S) > len(binary_N):
        leading_length = len(S) - len(binary_N)
        
        # Check if any leading fixed digits are 1
        for i in range(leading_length):
            if S[i] == '1':
                return -1  # S will always exceed N
        
        # Replace the ? in the leading positions with 0
        result = ''
        for i in range(leading_length):
            if S[i] == '?':
                result += '0'
            else:  # S[i] == '0'
                result += S[i]
        
        # Continue with the regular logic for the remaining characters
        remaining_S = S[leading_length:]
    else:
        result = ''
        remaining_S = S
    
    # Case 3: Compare character by character
    for i in range(len(remaining_S)):
        if remaining_S[i] == '?':
            if binary_N[i] == '1':
                result += '1'  # Replace ? with 1 when N also has 1
            else:  # binary_N[i] == '0'
                result += '0'  # Replace ? with 0 when N has 0
                # The rest of the ? can be replaced with 1s to maximize the value
                result += remaining_S[i+1:].replace('?', '1')
                return int(result, 2)
        elif remaining_S[i] == binary_N[i]:
            result += remaining_S[i]  # Digit matches N
        elif remaining_S[i] == '0' and binary_N[i] == '1':
            result += '0'  # S has 0, N has 1, so S is less than N at this position
            # The rest of the ? can be replaced with 1s
            result += remaining_S[i+1:].replace('?', '1')
            return int(result, 2)
        else:  # remaining_S[i] == '1' and binary_N[i] == '0'
            return -1  # S will always exceed N
    
    return int(result, 2)

S = input().strip()
N = int(input().strip())
print(solve(S, N))