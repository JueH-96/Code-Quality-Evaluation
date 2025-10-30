def main():
    N = int(input().strip())
    S = input().strip()
    
    # Condition 1: N must be odd
    if N % 2 == 0:
        print("No")
        return
    
    # Middle index (1-based): (N+1)//2
    # In 0-based indexing, that's mid - 1.
    mid = (N + 1) // 2
    
    # Condition 2: The first (mid - 1) characters must be '1'
    # Condition 3: The mid-th character (0-based index mid-1) must be '/'
    # Condition 4: The remaining characters must be '2'
    
    # Check if the first (mid - 1) characters are '1'
    if any(ch != '1' for ch in S[:mid-1]):
        print("No")
        return
    
    # Check if the mid-th character is '/'
    if S[mid-1] != '/':
        print("No")
        return
    
    # Check if the remaining characters are '2'
    if any(ch != '2' for ch in S[mid:]):
        print("No")
        return
    
    # If all conditions are satisfied
    print("Yes")

# Do not forget to call main
main()