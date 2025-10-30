def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    total_sum = 0
    
    for L in range(N):
        unique_count = 0
        freq = {}
        
        for R in range(L, N):
            if A[R] not in freq:
                freq[A[R]] = 0
            if freq[A[R]] == 0:
                unique_count += 1
            freq[A[R]] += 1
            
            total_sum += unique_count
            
    print(total_sum)