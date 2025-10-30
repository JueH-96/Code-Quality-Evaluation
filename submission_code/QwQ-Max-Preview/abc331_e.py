import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    L = int(input[ptr])
    ptr +=1
    
    a = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    b = list(map(int, input[ptr:ptr+M]))
    ptr +=M
    
    excluded = set()
    for _ in range(L):
        c = int(input[ptr])-1
        ptr +=1
        d = int(input[ptr])-1
        ptr +=1
        excluded.add((c, d))
    
    # Sort a and b in descending order with original indices
    sorted_a = sorted([(a[i], i) for i in range(N)], key=lambda x: (-x[0], x[1]))
    sorted_b = sorted([(b[j], j) for j in range(M)], key=lambda x: (-x[0], x[1]))
    
    heap = []
    visited = set()
    
    # Push the top pair (0,0)
    if N > 0 and M > 0:
        initial_sum = sorted_a[0][0] + sorted_b[0][0]
        heapq.heappush(heap, (-initial_sum, 0, 0))
        visited.add((0, 0))
    
    found = False
    while heap:
        current_neg_sum, i, j = heapq.heappop(heap)
        current_sum = -current_neg_sum
        
        a_orig = sorted_a[i][1]
        b_orig = sorted_b[j][1]
        
        if (a_orig, b_orig) not in excluded:
            print(current_sum)
            found = True
            break
        
        # Generate next pairs
        if i + 1 < N and (i+1, j) not in visited:
            new_sum = sorted_a[i+1][0] + sorted_b[j][0]
            heapq.heappush(heap, (-new_sum, i+1, j))
            visited.add((i+1, j))
        
        if j + 1 < M and (i, j+1) not in visited:
            new_sum = sorted_a[i][0] + sorted_b[j+1][0]
            heapq.heappush(heap, (-new_sum, i, j+1))
            visited.add((i, j+1))
    
    if not found:
        # According to problem statement, this is impossible
        pass

if __name__ == "__main__":
    main()