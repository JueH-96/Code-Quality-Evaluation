import sys
from collections import defaultdict

# Function to find the odd part and the exponent of 2
def get_odd_part_and_k(n):
    """Calculates the odd part O and the exponent k such that n = O * 2^k."""
    # Constraint S_i >= 1, so n will be positive.
    odd_part = n
    k = 0
    # Handle the case where n is 0 explicitly, although constraints prevent it.
    if n == 0:
        return 0, 0
    while odd_part % 2 == 0:
        odd_part //= 2
        k += 1
    return odd_part, k

def main():
    # Read the number of initial slime sizes
    N = int(sys.stdin.readline())
    
    # Use a dictionary to group slimes by their odd part.
    # The value associated with each odd part is another dictionary
    # mapping the exponent k (S = O * 2^k) to the total count of slimes
    # of that size.
    # defaultdict(dict) allows us to easily add counts: odd_part_map[odd_part][k] automatically
    # creates the inner dict and the key if they don't exist.
    odd_part_map = defaultdict(dict)
    
    # Read the initial slime sizes and counts
    for _ in range(N):
        S, C = map(int, sys.stdin.readline().split())
        
        # Find the odd part and exponent k for the current size S
        odd_part, k = get_odd_part_and_k(S)
        
        # Add the count C to the total count for this (odd_part, k) pair
        # Python handles arbitrary large integers for counts automatically
        odd_part_map[odd_part][k] = odd_part_map[odd_part].get(k, 0) + C

    # Initialize the total minimum number of remaining slimes
    total_remaining = 0

    # Process each distinct odd part independently
    for odd_part, counts_by_k in odd_part_map.items():
        # Initialize carried_count for this odd part. This represents slimes
        # produced by synthesizing slimes of size O * 2^(k-1) and now
        # contributing to the count of size O * 2^k.
        carried_count = 0
        
        # We need to process sizes O * 2^k in increasing order of k.
        # The smallest k we need to consider for this odd part is the minimum k
        # for which there's an initial count.
        # Since odd_part is a key in odd_part_map, counts_by_k is not empty.
        # Find the minimum initial k.
        min_k = min(counts_by_k.keys())
        current_k = min_k
        
        # Loop as long as there are slimes of size O * 2^current_k to process,
        # either from initial counts or carried over from O * 2^(current_k-1).
        # If carried_count is 0 and current_k is no longer in initial counts,
        # then there are no slimes of size O * 2^current_k or larger for this odd part,
        # so we can stop.
        while carried_count > 0 or current_k in counts_by_k:
            # Get the initial count for slimes of size O * 2^current_k
            initial_count = counts_by_k.get(current_k, 0)
            
            # The total number of slimes of size O * 2^current_k we can synthesize from
            # and possibly leave as remainder is the sum of initial count and carried count.
            current_total_count = initial_count + carried_count
            
            # The number of slimes of size O * 2^current_k that *cannot* be paired up
            # for synthesis is the remainder when dividing the total count by 2.
            # These slimes will remain at size O * 2^current_k in the final configuration.
            remaining_at_current_k = current_total_count % 2
            
            # The number of slimes of size O * 2^(current_k+1) produced is half the total count.
            # These slimes are carried over to the next size in the sequence for this odd part.
            carried_count = current_total_count // 2
            
            # Add the remaining slimes of size O * 2^current_k to the total minimum count
            total_remaining += remaining_at_current_k
            
            # Move to the next size in the sequence for this odd part
            current_k += 1

    # Print the final minimum number of slimes
    print(total_remaining)

if __name__ == "__main__":
    main()