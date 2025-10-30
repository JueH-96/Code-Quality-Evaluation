import sys
import heapq

# Function to encapsulate the logic for solving the problem
def solve():
    # Read the input string S from standard input
    S = sys.stdin.readline().strip()
    N = len(S)
    
    # Convert the string to a list of characters for efficient modifications
    # Python strings are immutable, so operations like replacement are costly.
    # A list allows O(1) time modification of characters by index.
    S_list = list(S)
    
    # Initialize a min-priority queue (min-heap).
    # It will store the starting indices of all occurrences of "WA" currently in the string.
    # Using a min-heap ensures that we can efficiently retrieve the smallest index,
    # which corresponds to the leftmost occurrence of "WA".
    pq = []
    
    # Perform an initial scan of the string to find all initial occurrences of "WA".
    # Add the starting index of each occurrence to the priority queue.
    for i in range(N - 1):
        if S_list[i] == 'W' and S_list[i+1] == 'A':
            heapq.heappush(pq, i)
            
    # Main loop: continue processing as long as there are "WA" occurrences in the priority queue.
    while pq:
        # Extract the smallest index `i` from the priority queue.
        # This `i` corresponds to the starting position of the leftmost "WA"
        # among all currently known "WA" occurrences.
        i = heapq.heappop(pq)
        
        # **Validity Check:** Before performing the replacement, we must check if the characters
        # at indices `i` and `i+1` still form "WA". This is necessary because previous
        # replacements might have modified the characters at these positions.
        # If `S_list[i]` is not 'W' or `S_list[i+1]` is not 'A', then the index `i`
        # retrieved from the priority queue is "stale" (it no longer points to a "WA").
        # In such cases, we simply ignore this index and continue to the next iteration.
        # We also check `i >= N - 1` as a safeguard, although logically indices should be within bounds.
        # The list `S_list` length `N` does not change.
        if i >= N - 1 or S_list[i] != 'W' or S_list[i+1] != 'A':
            continue
        
        # If the validity check passes, perform the replacement: "WA" -> "AC".
        # Modify the list of characters in place.
        S_list[i] = 'A'
        S_list[i+1] = 'C'
        
        # **Neighbor Check:** After replacing "WA" at index `i` with "AC",
        # it's possible that a new "WA" substring is formed starting at index `i-1`.
        # This happens if the character at `i-1` was 'W'.
        # The original sequence at `i-1, i, i+1` would have been 'W', 'W', 'A'.
        # After replacement, `S_list[i]` becomes 'A', so the sequence at `i-1, i` becomes 'W', 'A'.
        # Check if `i > 0` to ensure `i-1` is a valid index.
        if i > 0 and S_list[i-1] == 'W':
            # If a new "WA" is formed at index `i-1`, add `i-1` to the priority queue.
            # The heapq allows duplicate elements; if `i-1` is already present, it will be added again.
            # The validity check at the beginning of the loop handles stale entries effectively.
            heapq.heappush(pq, i-1)

    # After the loop terminates, no more "WA" substrings exist that can be formed or processed.
    # Join the characters in the list back into a single string.
    final_S = "".join(S_list)
    
    # Print the resulting string to standard output.
    print(final_S)

# Execute the solve function when the script is run
# This check ensures the code runs only when executed as a script, not when imported as a module.
if __name__ == '__main__':
    solve()