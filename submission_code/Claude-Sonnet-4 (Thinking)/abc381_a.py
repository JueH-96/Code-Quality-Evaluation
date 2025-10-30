N = int(input())
S = input()

if N % 2 == 0:
    print("No")
else:
    mid_idx = N // 2  # 0-indexed middle position
    
    # Check all conditions
    if (S[mid_idx] == '/' and 
        all(c == '1' for c in S[:mid_idx]) and 
        all(c == '2' for c in S[mid_idx+1:])):
        print("Yes")
    else:
        print("No")