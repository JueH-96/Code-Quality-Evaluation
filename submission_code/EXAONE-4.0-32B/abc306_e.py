import heapq
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    k = int(data[1])
    q = int(data[2])
    arr = [0] * (n + 1)
    in_heap = []
    out_heap = []
    sum_in = 0

    for i in range(1, n + 1):
        heapq.heappush(out_heap, (0, i))

    for _ in range(k):
        while out_heap and out_heap[0][0] != -arr[out_heap[0][1]]:
            heapq.heappop(out_heap)
        if out_heap:
            neg_val, idx = heapq.heappop(out_heap)
            val = -neg_val
            heapq.heappush(in_heap, (val, idx))
            sum_in += val

    outputs = []
    index_ptr = 3
    for _ in range(q):
        x = int(data[index_ptr])
        y = int(data[index_ptr + 1])
        index_ptr += 2
        old = arr[x]
        arr[x] = y
        heapq.heappush(out_heap, (-y, x))

        while in_heap and in_heap[0][0] != arr[in_heap[0][1]]:
            heapq.heappop(in_heap)
        while out_heap and out_heap[0][0] != -arr[out_heap[0][1]]:
            heapq.heappop(out_heap)

        while in_heap and len(in_heap) > k:
            while in_heap and in_heap[0][0] != arr[in_heap[0][1]]:
                heapq.heappop(in_heap)
            if len(in_heap) <= k:
                break
            val, idx = heapq.heappop(in_heap)
            heapq.heappush(out_heap, (-val, idx))
            sum_in -= val

        while in_heap and len(in_heap) < k and out_heap:
            while out_heap and out_heap[0][0] != -arr[out_heap[0][1]]:
                heapq.heappop(out_heap)
            if not out_heap:
                break
            neg_val, idx = heapq.heappop(out_heap)
            val = -neg_val
            heapq.heappush(in_heap, (val, idx))
            sum_in += val

        while in_heap and out_heap:
            while in_heap and in_heap[0][0] != arr[in_heap[0][1]]:
                heapq.heappop(in_heap)
            while out_heap and out_heap[0][0] != -arr[out_heap[0][1]]:
                heapq.heappop(out_heap)
            if not in_heap or not out_heap:
                break
            if in_heap[0][0] < -out_heap[0][0]:
                val_in, idx_in = heapq.heappop(in_heap)
                neg_val_out, idx_out = heapq.heappop(out_heap)
                val_out = -neg_val_out
                sum_in = sum_in - val_in + val_out
                heapq.heappush(in_heap, (val_out, idx_out))
                heapq.heappush(out_heap, (-val_in, idx_in))
            else:
                break
        outputs.append(str(sum_in))
    
    print("
".join(outputs))

if __name__ == "__main__":
    main()