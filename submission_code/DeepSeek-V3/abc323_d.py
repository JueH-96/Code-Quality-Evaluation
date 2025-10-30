import sys
import heapq

def main():
    N = int(sys.stdin.readline())
    slimes = []
    for _ in range(N):
        S, C = map(int, sys.stdin.readline().split())
        slimes.append((S, C))
    
    # Sort slimes by size in ascending order
    slimes.sort()
    
    # Use a dictionary to keep track of the counts of each size
    size_counts = {}
    for S, C in slimes:
        size_counts[S] = C
    
    # Use a priority queue to process the smallest sizes first
    heap = []
    for S in size_counts:
        heapq.heappush(heap, S)
    
    while heap:
        current_size = heapq.heappop(heap)
        if current_size not in size_counts:
            continue
        count = size_counts[current_size]
        if count < 2:
            continue
        # Perform synthesis
        new_size = 2 * current_size
        if new_size in size_counts:
            size_counts[new_size] += count // 2
        else:
            size_counts[new_size] = count // 2
        size_counts[current_size] = count % 2
        if size_counts[current_size] == 0:
            del size_counts[current_size]
        heapq.heappush(heap, new_size)
    
    # Calculate the total number of slimes
    total = 0
    for S in size_counts:
        total += size_counts[S]
    print(total)

if __name__ == "__main__":
    main()