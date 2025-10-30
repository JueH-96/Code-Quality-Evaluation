# Read input from stdin
N = int(input())
S = input()

# Check if N is even
if N % 2 == 0:
    print("No")
else:
    # Compute the middle index and lengths of the parts
    mid_idx = (N - 1) // 2
    len_first = mid_idx
    len_last = N - mid_idx - 1
    
    # Check the three conditions for an 11/22 string
    if all(c == '1' for c in S[:len_first]) and S[mid_idx] == '/' and all(c == '2' for c in S[N - len_last : N]):
        print("Yes")
    else:
        print("No")