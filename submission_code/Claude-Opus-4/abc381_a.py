# YOUR CODE HERE
N = int(input())
S = input()

# Check if length is odd
if N % 2 == 0:
    print("No")
else:
    # Find the middle position (1-indexed)
    middle = (N + 1) // 2
    
    # Check if middle character is '/'
    if S[middle - 1] != '/':
        print("No")
    else:
        # Check all characters before middle are '1'
        valid = True
        for i in range(middle - 1):
            if S[i] != '1':
                valid = False
                break
        
        # Check all characters after middle are '2'
        if valid:
            for i in range(middle, N):
                if S[i] != '2':
                    valid = False
                    break
        
        if valid:
            print("Yes")
        else:
            print("No")