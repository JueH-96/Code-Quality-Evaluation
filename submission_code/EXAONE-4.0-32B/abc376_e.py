import heapq
import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); index += 1
        k = int(data[index]); index += 1
        A = list(map(int, data[index:index+n]))
        index += n
        B = list(map(int, data[index:index+n]))
        index += n
        
        arr = list(zip(A, B))
        arr.sort(key=lambda x: x[0])
        
        heap_in = []
        heap_out = []
        sum_in = 0
        count_in = 0
        ans = float('inf')
        
        i = 0
        while i < n:
            j = i
            group = []
            while j < n and arr[j][0] == arr[i][0]:
                group.append(arr[j][1])
                j += 1
            x = arr[i][0]
            
            for b in group:
                if count_in < k:
                    heapq.heappush(heap_in, -b)
                    sum_in += b
                    count_in += 1
                else:
                    top_in = -heap_in[0]
                    if b < top_in:
                        removed_neg = heapq.heappop(heap_in)
                        removed = -removed_neg
                        sum_in -= removed
                        heapq.heappush(heap_in, -b)
                        sum_in += b
                        heapq.heappush(heap_out, removed)
                    else:
                        heapq.heappush(heap_out, b)
                        
            if count_in < k:
                i = j
                continue
                
            m = min(group)
            M = -heap_in[0]
            if m <= M:
                candidate = x * sum_in
            else:
                candidate = x * (sum_in - M + m)
                
            if candidate < ans:
                ans = candidate
                
            i = j
            
        results.append(str(ans))
    
    print("
".join(results))

if __name__ == "__main__":
    main()