# YOUR CODE HERE
S = input().strip()
N = int(input().strip())

# Check if the minimum possible value (all '?' replaced with '0') is > N
min_val_str = S.replace('?', '0')
min_val = int(min_val_str, 2)
if min_val > N:
    print(-1)
else:
    # Greedily try to maximize the value by setting '?' to '1' when possible
    result = list(S)
    for i in range(len(S)):
        if result[i] == '?':
            # Try setting this '?' to '1'
            temp = result.copy()
            temp[i] = '1'
            # Set all remaining '?' to '0' to get the minimum possible value
            for j in range(i+1, len(S)):
                if temp[j] == '?':
                    temp[j] = '0'
            if int(''.join(temp), 2) <= N:
                # Safe to set to '1'
                result[i] = '1'
            else:
                # Must set to '0'
                result[i] = '0'
    
    print(int(''.join(result), 2))