import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    events = []
    index = 2
    for _ in range(m):
        t = int(data[index])
        w = int(data[index+1])
        s = int(data[index+2])
        events.append((t, w, s))
        index += 3
    
    # Initialize the heap with (return_time, person_index)
    heap = []
    for i in range(1, n + 1):
        heapq.heappush(heap, (0, i))
    
    results = [0] * (n + 1)
    
    for t, w, s in events:
        while heap:
            current_return, current_index = heapq.heappop(heap)
            if current_return > t:
                break
            results[current_index] += w
            new_return = t + s
            heapq.heappush(heap, (new_return, current_index))
            break  # Only assign once per event
    
    for i in range(1, n + 1):
        print(results[i])

if __name__ == "__main__":
    main()