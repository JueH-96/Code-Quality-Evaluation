# YOUR CODE HERE
def count_arithmetic_subsequences():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        # If there's only one element, there's exactly one subsequence (the element itself)
        print(1)
        return
    
    count = 0
    l = 0
    
    while l < N:
        r = l
        if r + 1 < N:
            d = A[r + 1] - A[r]
            while r + 1 < N and A[r + 1] - A[r] == d:
                r += 1
        # Count all subsequences from l to r
        length = r - l + 1
        count += (length * (length + 1)) // 2
        l = r + 1
    
    print(count)