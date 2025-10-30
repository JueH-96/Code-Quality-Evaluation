import heapq
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    arr = [int(next(it)) for _ in range(n)]
    freq = [0] * (n + 1)
    
    for a in arr:
        if 0 <= a <= n:
            freq[a] += 1
            
    heap = []
    for i in range(n + 1):
        if freq[i] == 0:
            heapq.heappush(heap, i)
            
    out_lines = []
    for _ in range(q):
        idx = int(next(it)) - 1
        x = int(next(it))
        old_val = arr[idx]
        arr[idx] = x
        
        if 0 <= old_val <= n:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                heapq.heappush(heap, old_val)
                
        if 0 <= x <= n:
            freq[x] += 1
            
        while heap and freq[heap[0]] != 0:
            heapq.heappop(heap)
            
        mex = heap[0]
        out_lines.append(str(mex))
        
    print("
".join(out_lines))

if __name__ == "__main__":
    main()