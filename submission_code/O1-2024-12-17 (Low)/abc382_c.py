def main():
    import sys
    import heapq
    
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))
    
    # Pair each person with their gourmet level (A_i), keeping track of original index (i)
    # We'll use 0-based indexing internally, but add +1 for final output
    people = [(A[i], i) for i in range(N)]
    # Sort people by gourmet level ascending
    people.sort(key=lambda x: x[0])
    
    # Pair each sushi with its deliciousness (B_j), keeping track of its index (j)
    sushis = [(B[j], j) for j in range(M)]
    # Sort sushi by deliciousness ascending
    sushis.sort(key=lambda x: x[0])
    
    # We'll maintain a min-heap of people indices who qualify (A_i <= current sushi's B)
    # so we can pick the one with the smallest index i each time
    heap = []
    p = 0  # pointer over sorted people
    answers = [-1]*M
    
    # Go through sushis in ascending order of B
    for deliciousness, sushi_idx in sushis:
        # Add all people whose A_i <= current sushi's deliciousness
        while p < N and people[p][0] <= deliciousness:
            # push the person's original index onto the heap
            heapq.heappush(heap, people[p][1])
            p += 1
        
        # If our heap is non-empty, the top has the smallest i
        if heap:
            i_smallest = heap[0]
            # Store the (1-based) index of the person
            answers[sushi_idx] = i_smallest + 1
        else:
            # Nobody can eat this sushi
            answers[sushi_idx] = -1
    
    # Print the answers in the original sushi order
    print("
".join(map(str, answers)))


# Call main() at the end
if __name__ == "__main__":
    main()