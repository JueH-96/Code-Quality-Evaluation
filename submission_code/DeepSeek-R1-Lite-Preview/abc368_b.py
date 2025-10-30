import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create a max heap by pushing negative values
    heap = [-x for x in A if x > 0]
    heapq.heapify(heap)
    
    operations = 0
    while len(heap) >= 2:
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)
        first -= 1
        second -= 1
        if first > 0:
            heapq.heappush(heap, -first)
        if second > 0:
            heapq.heappush(heap, -second)
        operations += 1
    print(operations)

if __name__ == "__main__":
    main()