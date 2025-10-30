import heapq

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    C = [int(next(it)) for _ in range(N)]
    
    A.sort()
    C.sort()
    
    a_last = A[-1]
    c_last = C[-1]
    common = a_last + c_last
    a_c = a_last * c_last
    
    heap = []
    visited = set()
    
    for j in range(N):
        b_val = B[j]
        value = b_val * common + a_c
        heapq.heappush(heap, (-value, j, N-1, N-1))
        visited.add((j, N-1, N-1))
    
    count = 0
    ans = None
    while heap and count < K:
        neg_val, j, i, k = heapq.heappop(heap)
        current_val = -neg_val
        count += 1
        if count == K:
            ans = current_val
            break
            
        if i - 1 >= 0:
            state = (j, i-1, k)
            if state not in visited:
                visited.add(state)
                a_val = A[i-1]
                c_val = C[k]
                b_val = B[j]
                value = a_val * b_val + b_val * c_val + c_val * a_val
                heapq.heappush(heap, (-value, j, i-1, k))
                
        if k - 1 >= 0:
            state = (j, i, k-1)
            if state not in visited:
                visited.add(state)
                a_val = A[i]
                c_val = C[k-1]
                b_val = B[j]
                value = a_val * b_val + b_val * c_val + c_val * a_val
                heapq.heappush(heap, (-value, j, i, k-1))
                
    print(ans)

if __name__ == '__main__':
    main()