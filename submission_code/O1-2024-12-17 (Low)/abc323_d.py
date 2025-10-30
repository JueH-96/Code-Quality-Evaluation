def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Dictionary to keep track of how many slimes of each size exist
    slime_count = {}
    
    idx = 1
    for _ in range(N):
        size = int(input_data[idx]); idx += 1
        count = int(input_data[idx]); idx += 1
        slime_count[size] = slime_count.get(size, 0) + count
    
    # Min-heap of sizes we need to process
    heap = list(slime_count.keys())
    heapq.heapify(heap)

    while heap:
        s = heapq.heappop(heap)
        
        # If this size has been merged down to 0 in the meantime, skip
        c = slime_count[s]
        if c == 0:
            continue
        
        # Number of pairs we can merge
        merges = c // 2
        # Leftover after merging
        leftover = c % 2
        
        # Update current size with leftover
        slime_count[s] = leftover
        
        if merges > 0:
            new_s = 2 * s
            old_count = slime_count.get(new_s, 0)
            new_count = old_count + merges
            slime_count[new_s] = new_count
            
            # Push new_s back to the heap to try further merges
            # (only if we actually formed new slimes)
            heapq.heappush(heap, new_s)
    
    # Sum up the leftover slimes in the dictionary
    answer = sum(slime_count.values())
    print(answer)

# Call main() to conform to the required format.
if __name__ == "__main__":
    main()