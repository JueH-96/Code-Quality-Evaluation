import sys
import heapq

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); k = int(data[index+1]); index += 2
        A = list(map(int, data[index:index+n])); index += n
        B = list(map(int, data[index:index+n])); index += n
        
        if k == 0:
            results.append("0")
            continue
        
        pairs = []
        for i in range(n):
            pairs.append((A[i], B[i]))
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        
        groups = []
        if n > 0:
            current_a = pairs_sorted[0][0]
            current_list = [pairs_sorted[0][1]]
            for i in range(1, n):
                a, b = pairs_sorted[i]
                if a == current_a:
                    current_list.append(b)
                else:
                    groups.append((current_a, current_list))
                    current_a = a
                    current_list = [b]
            groups.append((current_a, current_list))
        else:
            groups = []
        
        heap = []
        current_sum = 0
        total_count = 0
        ans = 10**18
        
        for a_i, list_B in groups:
            min_group = min(list_B) if list_B else 0
            current_group_count = 0
            for b in list_B:
                if len(heap) < k:
                    heapq.heappush(heap, (-b, a_i))
                    current_sum += b
                    current_group_count += 1
                else:
                    if b < -heap[0][0]:
                        popped_val, group_id = heapq.heappop(heap)
                        popped_val_real = -popped_val
                        current_sum -= popped_val_real
                        if group_id == a_i:
                            current_group_count -= 1
                        heapq.heappush(heap, (-b, a_i))
                        current_sum += b
                        current_group_count += 1
            total_count += len(list_B)
            if total_count >= k:
                if current_group_count > 0:
                    candidate = a_i * current_sum
                else:
                    if heap:
                        largest_in_heap = -heap[0][0]
                        candidate = a_i * (current_sum - largest_in_heap + min_group)
                    else:
                        candidate = 10**18
                if candidate < ans:
                    ans = candidate
        results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()