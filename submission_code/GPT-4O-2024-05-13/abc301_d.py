# YOUR CODE HERE
def max_binary_value(S, N):
    # Replace all '?' with '1' to get the maximum possible value
    max_possible = int(S.replace('?', '1'), 2)
    
    # If the maximum possible value is less than or equal to N, return it
    if max_possible <= N:
        print(max_possible)
        return
    
    # Otherwise, we need to find the largest value less than or equal to N
    # by replacing '?' with '0' or '1'
    result = -1
    for i in range(2 ** S.count('?')):
        bin_str = bin(i)[2:].zfill(S.count('?'))
        temp_str = list(S)
        bin_index = 0
        for j in range(len(temp_str)):
            if temp_str[j] == '?':
                temp_str[j] = bin_str[bin_index]
                bin_index += 1
        value = int(''.join(temp_str), 2)
        if value <= N:
            result = max(result, value)
    
    print(result)

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
N = int(data[1])
max_binary_value(S, N)