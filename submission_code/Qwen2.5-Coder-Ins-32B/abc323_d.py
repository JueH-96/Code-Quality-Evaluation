# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    import heapq

    input = sys.stdin.read().split()
    N = int(input[0])
    slimes = []
    index = 1
    for _ in range(N):
        size = int(input[index])
        count = int(input[index + 1])
        slimes.append((size, count))
        index += 2

    # Sort slimes by size
    slimes.sort()

    # Use a min-heap to keep track of the counts of slimes
    min_heap = []
    for _, count in slimes:
        heapq.heappush(min_heap, count)

    while len(min_heap) > 1:
        # Pop two smallest counts
        count1 = heapq.heappop(min_heap)
        count2 = heapq.heappop(min_heap)
        
        # If both counts are odd, we can only merge one pair and leave one slime
        if count1 % 2 == 1 and count2 % 2 == 1:
            heapq.heappush(min_heap, 1)
            count1 -= 1
            count2 -= 1
        
        # Merge pairs
        merged_count = (count1 // 2) + (count2 // 2)
        if merged_count > 0:
            heapq.heappush(min_heap, merged_count)
        
        # If there are any remaining slimes, push them back
        if count1 % 2 == 1:
            heapq.heappush(min_heap, 1)
        if count2 % 2 == 1:
            heapq.heappush(min_heap, 1)

    # The number of slimes left in the heap is the answer
    print(len(min_heap))

if __name__ == "__main__":
    main()