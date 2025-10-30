def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    B = list(map(int, data[N+1:]))
    
    # Sort toys by size in descending order
    A.sort(reverse=True)
    
    # Sort boxes by size in descending order
    B.sort(reverse=True)
    
    # We need to find the minimum x such that x >= max(A[i] - B[i] for i in range(N-1))
    # and also x >= A[N-1] (the smallest toy must fit in the new box)
    
    # Initialize x to be at least large enough to fit the smallest toy
    min_x = A[-1]
    
    # Check if it's possible to fit all toys in the boxes
    # We will use a greedy approach to try to fit the largest toy in the largest box available
    # and so forth, if any toy cannot be placed in any box, then it's impossible
    
    # We need to check if the largest toy can fit in any box, if not, return -1 immediately
    if A[0] > B[0]:
        print(-1)
        return
    
    # We need to ensure that every toy can be placed in some box
    # We use a two-pointer technique to match toys to boxes
    j = 0  # Pointer for boxes
    for i in range(N-1):
        # Find the smallest box that can fit toy A[i]
        while j < N-1 and A[i] > B[j]:
            j += 1
        if j == N-1:
            # If we exhausted all boxes and the last toy doesn't fit in the last box
            # we need the new box to fit this toy
            min_x = max(min_x, A[i])
        else:
            # Otherwise, we fit A[i] in B[j] and move to the next box
            j += 1
    
    print(min_x)