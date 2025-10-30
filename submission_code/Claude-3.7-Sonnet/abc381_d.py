def max_1122_sequence_length(N, A):
    max_length = 0
    
    for start in range(N):
        frequency = {}
        
        for end in range(start, N):
            # Update frequency for the current element
            freq = frequency.get(A[end], 0) + 1
            frequency[A[end]] = freq
            
            # Check if any frequency exceeds 2, break if so
            if freq > 2:
                break
            
            # Check conditions
            if (end - start + 1) % 2 == 0:  # Condition 1: Even length
                valid = True
                
                # Condition 2: Adjacent pairs are the same
                for i in range(start, end + 1, 2):
                    if i + 1 <= end and A[i] != A[i+1]:
                        valid = False
                        break
                
                if valid:
                    # Condition 3: Each positive integer appears exactly twice
                    all_freq_2 = True
                    for count in frequency.values():
                        if count != 2:
                            all_freq_2 = False
                            break
                    
                    if all_freq_2:
                        max_length = max(max_length, end - start + 1)
    
    return max_length

# Read input
N = int(input())
A = list(map(int, input().split()))

# Print result
print(max_1122_sequence_length(N, A))