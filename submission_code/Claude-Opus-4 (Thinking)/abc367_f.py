# YOUR CODE HERE
N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

for _ in range(Q):
    l, r, L, R = map(int, input().split())
    # Convert to 0-indexed
    l -= 1
    L -= 1
    
    # Extract subsequences
    subseq_A = A[l:r]
    subseq_B = B[L:R]
    
    # Check if they can be rearranged to match
    # Two sequences can be rearranged to match if they have the same elements with same frequencies
    if sorted(subseq_A) == sorted(subseq_B):
        print("Yes")
    else:
        print("No")