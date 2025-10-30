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
    
    forbidden = set()
    for _ in range(L):
        c = int(input[ptr])
        ptr += 1
        d = int(input[ptr])
        ptr += 1
        forbidden.add((c, d))
    
    sorted_a = sorted(enumerate(a, 1), key=lambda x: (-x[1], x[0]))
    sorted_b = sorted(enumerate(b, 1), key=lambda x: (-x[1], x[0]))
    
    heap = []
    visited = set()
    i = 0
    j = 0
    initial_sum = sorted_a[i][1] + sorted_b[j][1]
    heapq.heappush(heap, (-initial_sum, i, j))
    visited.add((i, j))
    
    while heap:
        neg_sum, i, j = heapq.heappop(heap)
        current_sum = -neg_sum
        a_idx = sorted_a[i][0]
        b_idx = sorted_b[j][0]
        if (a_idx, b_idx) not in forbidden:
            print(current_sum)
            return
        
        if i + 1 < N and (i + 1, j) not in visited:
            new_sum = sorted_a[i + 1][1] + sorted_b[j][1]
            heapq.heappush(heap, (-new_sum, i + 1, j))
            visited.add((i + 1, j))
        
        if j + 1 < M and (i, j + 1) not in visited:
            new_sum = sorted_a[i][1] + sorted_b[j + 1][1]
            heapq.heappush(heap, (-new_sum, i, j + 1))
            visited.add((i, j + 1))
    
    print(0)

if __name__ == '__main__':
    main()