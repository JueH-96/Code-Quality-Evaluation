import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    L = int(data[idx+2])
    idx +=3
    
    a = list(map(int, data[idx:idx+N]))
    idx +=N
    
    b = list(map(int, data[idx:idx+M]))
    idx +=M
    
    forbidden = set()
    for _ in range(L):
        c = int(data[idx])
        d = int(data[idx+1])
        forbidden.add((c, d))
        idx +=2
    
    # Sort main dishes in descending order of cost, keeping original indices (1-based)
    sorted_main = sorted([(a[i], i+1) for i in range(N)], key=lambda x: (-x[0], x[1]))
    # Sort side dishes in descending order of cost, keeping original indices (1-based)
    sorted_side = sorted([(b[i], i+1) for i in range(M)], key=lambda x: (-x[0], x[1]))
    
    heap = []
    visited = set()
    
    # Initialize the heap with the first elements if available
    if sorted_main and sorted_side:
        i, j = 0, 0
        sum_val = sorted_main[i][0] + sorted_side[j][0]
        heap.append((-sum_val, i, j))
        visited.add((i, j))
    
    heapq.heapify(heap)
    
    found = False
    answer = 0
    
    while heap:
        neg_sum, i, j = heapq.heappop(heap)
        current_sum = -neg_sum
        
        original_main = sorted_main[i][1]
        original_side = sorted_side[j][1]
        
        if (original_main, original_side) not in forbidden:
            print(current_sum)
            found = True
            break
        
        # Generate next candidates
        for dx, dy in [(1, 0), (0, 1)]:
            ni = i + dx
            nj = j + dy
            if ni < len(sorted_main) and nj < len(sorted_side) and (ni, nj) not in visited:
                visited.add((ni, nj))
                new_sum = sorted_main[ni][0] + sorted_side[nj][0]
                heapq.heappush(heap, (-new_sum, ni, nj))
    
    # According to the problem statement, at least one set meal is offered, so found must be True

if __name__ == "__main__":
    main()