def max_11_22_substring_length(S):
    n = len(S)
    max_length = 0
    
    for i in range(n):
        if S[i] == '/':
            # Check how many '1's are to the left
            left_count = 0
            for j in range(i - 1, -1, -1):
                if S[j] == '1':
                    left_count += 1
                else:
                    break
            
            # Check how many '2's are to the right
            right_count = 0
            for j in range(i + 1, n):
                if S[j] == '2':
                    right_count += 1
                else:
                    break
            
            # The maximum valid length is 2*min(left_count, right_count) + 1
            valid_length = 2 * min(left_count, right_count) + 1
            
            max_length = max(max_length, valid_length)
    
    return max_length

N = int(input())
S = input().strip()
print(max_11_22_substring_length(S))