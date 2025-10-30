import heapq
from collections import defaultdict

def solve():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        K = int(input[ptr+1])
        ptr += 2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        elements = sorted(zip(A, B), key=lambda x: x[0])
        A_sorted = [x[0] for x in elements]
        B_sorted = [x[1] for x in elements]
        
        # Precompute sum_k_smallest
        sum_k_smallest = [0] * N
        current_sum = 0
        heap = []
        for i in range(N):
            if len(heap) < K:
                heapq.heappush(heap, -B_sorted[i])
                current_sum += B_sorted[i]
            else:
                if B_sorted[i] < -heap[0]:
                    popped = -heapq.heappop(heap)
                    current_sum -= popped
                    heapq.heappush(heap, -B_sorted[i])
                    current_sum += B_sorted[i]
            if i < K - 1:
                sum_k_smallest[i] = current_sum
            else:
                sum_k_smallest[i] = current_sum
        
        # Precompute sum_k_minus_1_smallest if K > 1
        if K == 1:
            sum_k_minus_1_smallest = [0] * N  # dummy, not used
        else:
            sum_k_minus_1_smallest = [0] * N
            current_sum_k_minus_1 = 0
            heap_k_minus_1 = []
            for i in range(N):
                if len(heap_k_minus_1) < (K-1):
                    heapq.heappush(heap_k_minus_1, -B_sorted[i])
                    current_sum_k_minus_1 += B_sorted[i]
                else:
                    if (K-1) > 0 and B_sorted[i] < -heap_k_minus_1[0]:
                        popped = -heapq.heappop(heap_k_minus_1)
                        current_sum_k_minus_1 -= popped
                        heapq.heappush(heap_k_minus_1, -B_sorted[i])
                        current_sum_k_minus_1 += B_sorted[i]
                if i < (K-2):
                    sum_k_minus_1_smallest[i] = current_sum_k_minus_1
                else:
                    sum_k_minus_1_smallest[i] = current_sum_k_minus_1
        
        # Create index map
        index_map = defaultdict(list)
        for i in range(N):
            index_map[A_sorted[i]].append(i)
        
        min_product = float('inf')
        
        # Iterate over each unique x in sorted order
        unique_x = sorted(index_map.keys())
        for x in unique_x:
            indices = index_map[x]
            s = indices[0]
            e = indices[-1]
            
            # Check if group has at least one element
            if (e - s + 1) < 1:
                continue
            
            # Check if up to e, there are enough elements
            if e < (K - 1):
                continue
            
            # Calculate sum_a
            sum_a = sum_k_smallest[e]
            
            # Calculate sum_b
            sum_b = float('inf')
            if s > 0:
                if s >= (K - 1):
                    if (s - 1) >= 0:
                        sum_without = sum_k_minus_1_smallest[s - 1]
                        min_B_in_G = min(B_sorted[s:e+1])
                        sum_b = sum_without + min_B_in_G
            
            # Collect valid candidates
            candidates = []
            if sum_a != 0:
                candidates.append(sum_a)
            if sum_b != float('inf'):
                candidates.append(sum_b)
            
            if not candidates:
                continue
            
            minimal_sum = min(candidates)
            product = x * minimal_sum
            if product < min_product:
                min_product = product
        
        if min_product == float('inf'):
            print(0)
        else:
            print(min_product)

if __name__ == '__main__':
    solve()