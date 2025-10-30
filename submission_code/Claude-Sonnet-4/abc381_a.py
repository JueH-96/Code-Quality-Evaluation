# YOUR CODE HERE
N = int(input())
S = input()

# Check if length is odd
if N % 2 == 0:
    print("No")
else:
    # Calculate middle position (1-indexed)
    mid = (N + 1) // 2
    
    # Check if middle character is '/'
    if S[mid - 1] != '/':  # Convert to 0-indexed
        print("No")
    else:
        # Check first part (should all be '1')
        first_part_valid = True
        for i in range(mid - 1):  # 0 to mid-2 (0-indexed)
            if S[i] != '1':
                first_part_valid = False
                break
        
        # Check second part (should all be '2')
        second_part_valid = True
        for i in range(mid, N):  # mid to N-1 (0-indexed)
            if S[i] != '2':
                second_part_valid = False
                break
        
        if first_part_valid and second_part_valid:
            print("Yes")
        else:
            print("No")