import sys
import heapq

# Using sys.stdin.readline for faster input, and sys.stdout.write for faster output.
# sys.stdin.readline() reads the line along with the trailing newline character.
# .strip() removes whitespace characters from the beginning and end, including the newline.
# sys.stdout.write() does not automatically add a newline, so it must be added manually.

def solve():
    S = sys.stdin.readline().strip()
    n = len(S)
    
    if n < 2: # "WA" substring needs at least 2 characters.
        sys.stdout.write(S + "
")
        return
        
    s_list = list(S) # Convert string to list of characters for mutable operations
    
    # Initialize a min-priority queue (heap) with the starting indices of all initial "WA" occurrences.
    # We want to process the leftmost "WA" first, hence a min-heap on indices.
    initial_wa_indices = []
    for i in range(n - 1): # Iterate up to n-2 (i.e., S[n-2]S[n-1] is the last pair)
        if s_list[i] == 'W' and s_list[i+1] == 'A':
            initial_wa_indices.append(i)
    
    # heapq.heapify transforms the list into a heap in-place. This is O(K) time,
    # where K is the number of initial "WA"s (K <= N).
    heapq.heapify(initial_wa_indices)
    pq = initial_wa_indices # pq now refers to the heapified list
            
    # Loop as long as there are candidate "WA"s in the priority queue
    while pq:
        idx = heapq.heappop(pq) # Get the smallest index (leftmost "WA" candidate)
        
        # Critical check: Verify that S[idx]S[idx+1] is still "WA".
        # It's possible that a previous operation has modified S[idx] or S[idx+1],
        # invalidating this "WA" candidate.
        if not (s_list[idx] == 'W' and s_list[idx+1] == 'A'):
            continue # This "WA" is no longer valid, skip to the next candidate
            
        # Perform the replacement: "WA" at S[idx]S[idx+1] becomes "AC"
        s_list[idx] = 'A'
        s_list[idx+1] = 'C'
        
        # Check if a new "WA" has formed at S[idx-1]S[idx] due to this replacement.
        # This occurs if S[idx-1] was 'W' and S[idx] (which was 'W') became 'A'.
        # Original state: S[idx-1] S[idx] S[idx+1] was S[idx-1]'W''A'
        # After replacement: S[idx-1] S[idx] S[idx+1] becomes S[idx-1]'A''C'
        # If S[idx-1] is 'W', then S[idx-1]S[idx] is now 'WA'.
        if idx > 0: # Ensure idx-1 is a valid index
            if s_list[idx-1] == 'W': # Check if S[idx-1] is 'W'
                                     # s_list[idx] is now 'A' due to the replacement above
                # The pair S[idx-1]S[idx] is now "WA".
                # This "WA" at idx-1 could not have existed before this operation,
                # because S[idx] was 'W', making S[idx-1]S[idx] be S[idx-1]'W', not "WA".
                # Add its starting index (idx-1) to the priority queue.
                heapq.heappush(pq, idx - 1)
                
    # After all operations, join the list of characters back into a string and print.
    sys.stdout.write("".join(s_list) + "
")

# Standard boilerplate for competitive programming to call the main solve function.
if __name__ == "__main__":
    solve()