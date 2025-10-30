K = int(input())
S = input().strip()
T = input().strip()

n = len(S)
m = len(T)

if abs(n - m) > K:
    print("No")
else:
    # Initialize previous row for i=0
    start_j_prev = max(0, 0 - K)
    end_j_prev = min(m, 0 + K)
    prev_row = [float('inf')] * (2 * K + 1)
    for j in range(start_j_prev, end_j_prev + 1):
        prev_row[j - start_j_prev] = j  # insert j characters
    
    for i in range(1, n + 1):
        start_j = max(0, i - K)
        end_j = min(m, i + K)
        curr_row = [float('inf')] * (2 * K + 1)
        prev_start = max(0, (i - 1) - K)
        prev_end = min(m, (i - 1) + K)
        
        for j in range(start_j, end_j + 1):
            idx = j - start_j
            # Deletion
            if j >= prev_start and j <= prev_end:
                deletion = prev_row[j - prev_start] + 1
            else:
                deletion = float('inf')
            # Insertion
            if j - 1 >= start_j and j - 1 <= end_j:
                insertion = curr_row[j - 1 - start_j] + 1
            else:
                insertion = float('inf')
            # Replace
            if j - 1 >= prev_start and j - 1 <= prev_end:
                replace_cost = prev_row[j - 1 - prev_start] + (S[i-1] != T[j-1])
            else:
                replace_cost = float('inf')
            
            curr_val = min(deletion, insertion, replace_cost)
            if j == 0:
                curr_val = i  # All deletions
            curr_row[idx] = curr_val
        
        prev_row = curr_row
        min_val = min(prev_row)
        if min_val > K:
            print("No")
            exit()
    
    # Check the value at j = m
    start_j_final = max(0, n - K)
    end_j_final = min(m, n + K)
    if m < start_j_final or m > end_j_final:
        print("No")
    else:
        idx = m - start_j_final
        if idx >= len(prev_row):
            print("No")
        else:
            if prev_row[idx] <= K:
                print("Yes")
            else:
                print("No")