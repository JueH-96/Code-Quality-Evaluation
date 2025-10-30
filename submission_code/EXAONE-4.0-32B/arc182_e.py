import math
import sys
import bisect
import heapq
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    N = int(next(it)); M = int(next(it)); C = int(next(it)); K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    if C == 0:
        ans = K * min(A)
        print(ans)
        return
        
    g = math.gcd(C, M)
    P = M // g
    
    B = sorted(A)
    n = len(B)
    prefix_min = [0] * n
    suffix_min = [0] * n
    prefix_min[0] = B[0]
    for i in range(1, n):
        prefix_min[i] = min(prefix_min[i-1], B[i])
    suffix_min[n-1] = B[n-1]
    for i in range(n-2, -1, -1):
        suffix_min[i] = min(suffix_min[i+1], B[i])
    
    total_period_sum = 0
    total_period_sum += min(A)
    
    if P == 1:
        full_cycles = K
        rem = 0
        total_sum = full_cycles * total_period_sum
        print(total_sum)
        return
        
    if P <= 200000:
        for j in range(1, P):
            T_val = j * g
            threshold = M - T_val
            pos = bisect.bisect_left(B, threshold)
            candidate1 = 10**20
            candidate2 = 10**20
            if pos > 0:
                candidate1 = T_val + prefix_min[pos-1]
            if pos < n:
                candidate2 = T_val - M + suffix_min[pos]
            min_res = min(candidate1, candidate2)
            total_period_sum += min_res
    else:
        events_grouped = defaultdict(list)
        for a in A:
            j0 = (a // g) + 1
            if j0 < P:
                events_grouped[j0].append(a)
                
        event_js = sorted(events_grouped.keys())
        idx = bisect.bisect_left(B, g)
        left_min = B[0] if idx > 0 else 10**20
        heap_right = []
        removed = set()
        for i in range(idx, n):
            heapq.heappush(heap_right, B[i])
            
        intervals = sorted(set([1] + event_js + [P]))
        total_period_sum_rest = 0
        current = 1
        for j0_val in intervals:
            if current < j0_val:
                while heap_right and heap_right[0] in removed:
                    heapq.heappop(heap_right)
                right_min = heap_right[0] if heap_right else 10**20
                count = j0_val - current
                if left_min > 10**19 and right_min > 10**19:
                    seg = 0
                elif left_min > 10**19:
                    total_j = (current + j0_val - 1) * count // 2
                    seg = count * right_min - M * count + g * total_j
                elif right_min > 10**19:
                    total_j = (current + j0_val - 1) * count // 2
                    seg = count * left_min + g * total_j
                else:
                    if M + left_min <= right_min:
                        total_j = (current + j0_val - 1) * count // 2
                        seg = count * (M + left_min) + g * total_j
                    else:
                        total_j = (current + j0_val - 1) * count // 2
                        seg = count * right_min - M * count + g * total_j
                total_period_sum_rest += seg
                
            if j0_val in events_grouped:
                for a_val in events_grouped[j0_val]:
                    removed.add(a_val)
                    if a_val < left_min:
                        left_min = a_val
            current = j0_val
            
        if current <= P-1:
            count = P - current
            while heap_right and heap_right[0] in removed:
                heapq.heappop(heap_right)
            right_min = heap_right[0] if heap_right else 10**20
            if left_min > 10**19 and right_min > 10**19:
                seg = 0
            elif left_min > 10**19:
                total_j = (current + P - 1) * count // 2
                seg = count * right_min - M * count + g * total_j
            elif right_min > 10**19:
                total_j = (current + P - 1) * count // 2
                seg = count * left_min + g * total_j
            else:
                if M + left_min <= right_min:
                    total_j = (current + P - 1) * count // 2
                    seg = count * (M + left_min) + g * total_j
                else:
                    total_j = (current + P - 1) * count // 2
                    seg = count * right_min - M * count + g * total_j
            total_period_sum_rest += seg
            
        total_period_sum += total_period_sum_rest
        
    full_cycles = K // P
    rem = K % P
    total_sum = full_cycles * total_period_sum
    
    if rem > 0:
        total_sum += min(A)
        if rem > 1:
            for k_val in range(1, rem):
                T_val = (C * k_val) % M
                threshold = M - T_val
                pos = bisect.bisect_left(B, threshold)
                candidate1 = 10**20
                candidate2 = 10**20
                if pos > 0:
                    candidate1 = T_val + prefix_min[pos-1]
                if pos < n:
                    candidate2 = T_val - M + suffix_min[pos]
                min_res_k = min(candidate1, candidate2)
                total_sum += min_res_k
                
    print(total_sum)

if __name__ == '__main__':
    main()