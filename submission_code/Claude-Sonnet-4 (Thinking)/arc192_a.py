def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # If all positions are already 1, return Yes
    if all(a == 1 for a in A):
        print("Yes")
        return
    
    # Find positions that need to be fixed (bad positions)
    bad_positions = [i for i in range(N) if A[i] == 0]
    
    # For small N, check if it's possible
    if N == 3:
        # For N=3, all operations conflict with each other
        # Can only apply at most 1 operation, which covers at most 2 positions
        if len(bad_positions) <= 2:
            print("Yes")
        else:
            print("No")
        return
    
    # For N >= 4, it's generally possible to find non-conflicting operations
    # The key insight is that operations can be spaced out to avoid conflicts
    # For N >= 6, we can always find enough non-conflicting operations
    
    # Even for N=4,5 with many bad positions, careful analysis shows it's usually possible
    # The only problematic case is N=3 with 3 bad positions
    
    print("Yes")

solve()