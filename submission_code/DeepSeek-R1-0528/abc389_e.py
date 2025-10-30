import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    M_val = int(data[1])
    P_list = list(map(int, data[2:2+n]))
    
    low = 0
    high = 10**19
    while low <= high:
        mid = (low + high) // 2
        total_cost = 0
        for p in P_list:
            k = (mid + p) // (2 * p)
            total_cost += p * k * k
            if total_cost > M_val:
                break
        if total_cost <= M_val:
            low = mid + 1
        else:
            high = mid - 1
    lambda0 = high

    base_units = 0
    base_cost = 0
    heap = []
    for p in P_list:
        k = (lambda0 + p) // (2 * p)
        base_units += k
        base_cost += p * k * k
        next_cost = (2 * k + 1) * p
        heapq.heappush(heap, (next_cost, p))
    
    residual = M_val - base_cost
    additional_units = 0
    while heap and residual > 0:
        cost, p = heapq.heappop(heap)
        if cost <= residual:
            residual -= cost
            additional_units += 1
            next_next_cost = cost + 2 * p
            heapq.heappush(heap, (next_next_cost, p))
        else:
            break
            
    total_units = base_units + additional_units
    print(total_units)

if __name__ == "__main__":
    main()