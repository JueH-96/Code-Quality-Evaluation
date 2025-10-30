def max_binary_leq_N(S, N):
    # Convert N to binary string to compare lengths
    N_bin = bin(N)[2:]
    len_S = len(S)
    len_N = len(N_bin)
    
    # If S is longer than N's binary representation, the maximum possible is N if S can represent N
    if len_S > len_N:
        # Check if S can represent N
        # Replace all '?' with '0' and see if it's <= N
        min_S = S.replace('?', '0')
        min_val = int(min_S, 2)
        if min_val > N:
            return -1
        # Replace all '?' with '1' and see if it's <= N
        max_S = S.replace('?', '1')
        max_val = int(max_S, 2)
        if max_val <= N:
            return max_val
        # Otherwise, find the maximum possible <= N
        # We need to find the largest number <= N by replacing '?'s
        # We can do this by trying to set '?'s to '1' as much as possible
        # Start from the left and try to set '1' if possible
        result = list(S)
        for i in range(len_S):
            if result[i] == '?':
                # Try to set it to '1'
                result[i] = '1'
                temp = ''.join(result)
                temp_val = int(temp, 2)
                if temp_val > N:
                    # Set it back to '0'
                    result[i] = '0'
        final_val = int(''.join(result), 2)
        if final_val <= N:
            return final_val
        else:
            return -1
    elif len_S < len_N:
        # All possible values are less than N, so the maximum is the maximum possible value
        max_S = S.replace('?', '1')
        max_val = int(max_S, 2)
        return max_val
    else:
        # len_S == len_N
        # We need to find the maximum possible <= N
        # Start from the left and try to set '1' if possible
        result = list(S)
        for i in range(len_S):
            if result[i] == '?':
                # Try to set it to '1'
                result[i] = '1'
                temp = ''.join(result)
                temp_val = int(temp, 2)
                if temp_val > N:
                    # Set it back to '0'
                    result[i] = '0'
        final_val = int(''.join(result), 2)
        if final_val <= N:
            return final_val
        else:
            return -1

# Read input
S = input().strip()
N = int(input())

# Compute and print the result
print(max_binary_leq_N(S, N))