import heapq
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    q = int(next(it))
    
    cur = [0] * n
    in1 = [False] * n
    heap1 = []
    heap2 = []
    
    for i in range(k):
        heapq.heappush(heap1, (0, i))
        in1[i] = True
    for i in range(k, n):
        heapq.heappush(heap2, (0, i))
    
    total = 0
    out_lines = []
    
    for _ in range(q):
        x = int(next(it))
        y = int(next(it))
        idx = x - 1
        old_val = cur[idx]
        new_val = y
        cur[idx] = new_val
        
        if in1[idx]:
            total -= old_val
            in1[idx] = False
            gap = True
        else:
            gap = False
        
        heapq.heappush(heap2, (-new_val, idx))
        
        if gap:
            while heap2:
                neg_val, j = heapq.heappop(heap2)
                val = -neg_val
                if val != cur[j] or in1[j]:
                    continue
                heapq.heappush(heap1, (val, j))
                total += val
                in1[j] = True
                break
        else:
            while heap1:
                val, j = heap1[0]
                if val != cur[j] or not in1[j]:
                    heapq.heappop(heap1)
                else:
                    break
            if heap1:
                min_val = heap1[0][0]
                if new_val > min_val:
                    val_min, j_min = heapq.heappop(heap1)
                    total -= val_min
                    in1[j_min] = False
                    heapq.heappush(heap2, (-val_min, j_min))
                    heapq.heappush(heap1, (new_val, idx))
                    total += new_val
                    in1[idx] = True
        
        out_lines.append(str(total))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()