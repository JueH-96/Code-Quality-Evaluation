import heapq

def main():
    import sys
    data = sys.stdin.read().splitlines()
    Q = int(data[0])
    total_days = 0
    heap = []
    results = []
    index = 1
    for _ in range(Q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            heapq.heappush(heap, total_days)
        elif parts[0] == '2':
            T = int(parts[1])
            total_days += T
        elif parts[0] == '3':
            H = int(parts[1])
            threshold = total_days - H
            cnt = 0
            while heap and heap[0] <= threshold:
                heapq.heappop(heap)
                cnt += 1
            results.append(cnt)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()