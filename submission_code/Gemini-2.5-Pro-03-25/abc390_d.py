# YOUR CODE HERE
import sys
import bisect

# Setting a higher recursion depth limit might be necessary for larger N, 
# although N=12 is relatively small and might not exceed the default limit (often 1000).
# If you encounter a RecursionError, uncommenting the following line might help.
# sys.setrecursionlimit(2000) 

def solve():
    # Read the number of bags, N
    N = int(sys.stdin.readline())
    # Read the initial number of stones in each bag
    A = list(map(int, sys.stdin.readline().split()))
    
    # This set will store all unique XOR sums encountered.
    possible_xor_sums = set()
    
    # Memoization dictionary to store visited states.
    # A state is defined by (idx, current_sums_tuple), where idx is the index
    # of the element A[idx] being processed, and current_sums_tuple is a sorted
    # tuple of the sums of stones in the groups formed so far.
    # The value True indicates that the state has been processed.
    memo = {} 

    # Define the recursive function `generate`
    # idx: The index of the current element A[idx] from the input list A that we are considering.
    # current_sums_tuple: A tuple containing the sums of stones for each group in the partition
    #                     formed using elements A[0] through A[idx-1]. The tuple is kept sorted
    #                     to ensure a canonical representation for memoization.
    def generate(idx, current_sums_tuple):
        # Construct the state representation for memoization lookup.
        state = (idx, current_sums_tuple)
        
        # Check if this state has already been visited/processed. If so, return immediately
        # to avoid redundant computation (Dynamic Programming / Memoization).
        if state in memo:
             return
        
        # Base Case: If we have processed all N elements (indices 0 to N-1).
        if idx == N:
            # Calculate the final XOR sum for the partition represented by current_sums_tuple.
            current_xor_sum = 0
            for s in current_sums_tuple:
                current_xor_sum ^= s
            # Add the calculated XOR sum to the set of possible results.
            # Sets automatically handle uniqueness.
            possible_xor_sums.add(current_xor_sum)
            # Mark this state as processed.
            memo[state] = True 
            return

        # Get the value of the current element A[idx].
        current_val = A[idx]
        
        # ----- Explore Option 1: Start a new group with current_val -----
        # This corresponds to placing A[idx] into a new group by itself.
        # The number of groups in the partition increases by 1.
        
        # Create a mutable list from the current tuple of sums to modify it.
        new_sums_list_opt1 = list(current_sums_tuple)
        
        # Find the correct position to insert current_val to maintain the sorted order.
        # bisect.bisect_left gives the index in O(log k) time, where k is the number of groups.
        insert_pos_opt1 = bisect.bisect_left(new_sums_list_opt1, current_val)
        
        # Insert current_val into the list. This operation takes O(k) time.
        new_sums_list_opt1.insert(insert_pos_opt1, current_val)
        
        # Make the recursive call to process the next element (idx+1) with the updated tuple of sums.
        # Convert the list back to a tuple for hashability (required for dictionary keys / set elements).
        generate(idx + 1, tuple(new_sums_list_opt1))
        
        
        # ----- Explore Option 2: Add current_val to one of the existing groups -----
        # This corresponds to merging A[idx] into one of the existing groups.
        # The number of groups remains the same.
        
        # Iterate through each existing group (represented by its sum at index i).
        for i in range(len(current_sums_tuple)):
            # Calculate the new sum if current_val is added to the i-th group.
            new_sum = current_sums_tuple[i] + current_val
            
            # Prepare the list of sums for the next state in this branch.
            # It will contain all sums from current_sums_tuple except the one at index i,
            # plus the newly calculated sum `new_sum`.
            
            # Create a temporary list based on the current tuple. It's important to start
            # from the original tuple for each iteration `i` to avoid cumulative effects.
            next_sums_list_opt2 = list(current_sums_tuple)
            
            # Remove the original sum at index i. `del` on a list takes O(k) time.
            del next_sums_list_opt2[i] 
            
            # Insert the `new_sum` into this modified list while maintaining sorted order.
            # Find insertion point using bisect_left: O(log k)
            insert_pos_opt2 = bisect.bisect_left(next_sums_list_opt2, new_sum)
            # Insert into list: O(k)
            next_sums_list_opt2.insert(insert_pos_opt2, new_sum)

            # Make the recursive call to process the next element (idx+1) with this new tuple of sums.
            generate(idx + 1, tuple(next_sums_list_opt2))

        # After exploring all branches (Option 1 and all iterations of Option 2) stemming from the current state,
        # mark this state as visited/processed in the memoization table.
        memo[state] = True 

    # Initiate the recursive process. Start with the first element (index 0) and an empty partition (empty tuple of sums).
    generate(0, tuple())
    
    # The final answer is the total number of distinct XOR sums collected in the set.
    print(len(possible_xor_sums))

# Call the main function `solve` to run the program.
solve()