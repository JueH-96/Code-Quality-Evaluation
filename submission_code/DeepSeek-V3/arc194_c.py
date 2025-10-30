n = int(input())
A = list(map(int, input().split()))

if all(a == 1 for a in A):
    print("Yes")
else:
    # Check if there's at least one pair of adjacent 0s or the 0s can be covered in some way
    # The critical observation is that each 0 must be adjacent to another 0, forming a contiguous block of 0s of even length or overlapping.
    # Alternatively, the 0s must be in positions that can be covered by operations, which require that for every 0 at position i, either i and i+1 are 0, or i-1 and i are 0.
    has_adjacent_zeros = False
    for i in range(n):
        if A[i] == 0 and A[(i+1)%n] == 0:
            has_adjacent_zeros = True
            break
    if has_adjacent_zeros:
        print("Yes")
    else:
        print("No")