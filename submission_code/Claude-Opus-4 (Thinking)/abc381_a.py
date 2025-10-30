# YOUR CODE HERE
n = int(input())
s = input()

if n % 2 == 0:
    print("No")
else:
    mid = n // 2  # 0-indexed middle position
    
    # Check if it's a valid 11/22 string
    valid = (s[mid] == '/' and 
             all(c == '1' for c in s[:mid]) and 
             all(c == '2' for c in s[mid+1:]))
    
    print("Yes" if valid else "No")