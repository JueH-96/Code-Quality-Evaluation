import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    L = int(input[ptr])
    ptr += 1
    
    a = list(map(int, input[ptr:ptr+N]))
    ptr += N
    b = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    # Sort main dishes and side dishes in descending order, keeping original indices
    sorted_a = sorted([(a[i], i+1) for i in range(N)], key=lambda x: (-x[0], x[1]))
    sorted_b = sorted([(b[i], i+1) for i in range(M)], key=lambda x: (-x[0], x[1]))
    
    excluded = set()
    for _ in range(L):
        c = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        excluded.add((c, d))
    
    # Initialize heap and visited set
    heap = []
    visited = set()
    
    # Push the initial (0,0) pair
    initial_sum = sorted_a[0][0] + sorted_b[0][0]
    heapq.heappush(heap, (-initial_sum, 0, 0))
    visited.add((0, 0))
    
    while heap:
        current = heapq.heappop(heap)
        current_sum = -current[0]
        i, j = current[1], current[2]
        original_c = sorted_a[i][1]
        original_d = sorted_b[j][1]
        
        if (original_c, original_d) not in excluded:
            print(current_sum)
            return
        
        # Generate next candidates
        if i + 1 < len(sorted_a):
            ni, nj = i + 1, j
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                sum_next = sorted_a[ni][0] + sorted_b[nj][0]
                heapq.heappush(heap, (-sum_next, ni, nj))
        if j + 1 < len(sorted_b):
            ni, nj = i, j + 1
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                sum_next = sorted_a[ni][0] + sorted_b[nj][0]
                heapq.heappush(heap, (-sum_next, ni, nj))
    
    # The code should have returned before here, but just in case
    print(0)

if __name__ == "__main__":
    main()