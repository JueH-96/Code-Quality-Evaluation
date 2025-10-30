def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    
    # Condition 1: Check if length is odd
    if N % 2 == 0:
        print("No")
        return
    
    mid_index = (N + 1) // 2 - 1  # zero-indexed position for the '/'
    
    # Check if the middle character is '/'
    if S[mid_index] != '/':
        print("No")
        return
    
    # First part: from index 0 to mid_index - 1, all must be '1'
    for i in range(mid_index):
        if S[i] != '1':
            print("No")
            return
    
    # Second part: from index mid_index+1 to end, all must be '2'
    for i in range(mid_index + 1, N):
        if S[i] != '2':
            print("No")
            return
    
    # If all the conditions are satisfied:
    print("Yes")

if __name__ == '__main__':
    main()