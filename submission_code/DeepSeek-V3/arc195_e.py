import heapq

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    max_heap = []
    sum_s = 0
    res = 0
    operations = 0  # tracks the number of elements we can delete (operations is the count of elements in S)
    
    for num in A:
        if num >= 0:
            sum_s += num
            operations += 1
            heapq.heappush(max_heap, num)
        else:
            if max_heap:
                current_min = max_heap[0]
                if num > current_min:
                    sum_s += num - current_min
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, num)
        # The operations (number of elements in S) is len(max_heap)
    print(sum_s)

solve()