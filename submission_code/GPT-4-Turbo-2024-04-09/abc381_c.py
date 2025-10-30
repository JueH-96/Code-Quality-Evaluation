def max_1122_string_length(N, S):
    max_length = 0
    
    # Iterate through each character in the string
    i = 0
    while i < N:
        if S[i] == '/':
            # We found a '/', now count '1's backward and '2's forward
            count1 = 0
            count2 = 0
            
            # Count '1's backward
            j = i - 1
            while j >= 0 and S[j] == '1':
                count1 += 1
                j -= 1
            
            # Count '2's forward
            k = i + 1
            while k < N and S[k] == '2':
                count2 += 1
                k += 1
            
            # The valid 11/22 string length
            min_count = min(count1, count2)
            current_length = 1 + 2 * min_count
            
            # Update the maximum length found
            max_length = max(max_length, current_length)
        
        i += 1
    
    return max_length

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

print(max_1122_string_length(N, S))