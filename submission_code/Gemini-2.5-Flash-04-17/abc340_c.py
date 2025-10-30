import sys
from collections import Counter

# Read input
N = int(sys.stdin.readline())

# Algorithm using iterative expansion with counts
# counts: Dictionary mapping number value >= 2 to its total occurrences on the blackboard
# to_process: Set of distinct number values >= 2 that are currently on the blackboard and need processing in the current iteration
# total_cost: Accumulated cost

counts = Counter()
to_process = set()
total_cost = 0

# Initialize with the starting number N, if it's >= 2
if N >= 2:
    counts[N] = 1
    to_process.add(N)

# Iteratively process numbers level by level.
# Each iteration of the outer while loop processes all distinct numbers that were
# generated in the previous step and adds newly generated numbers >= 2 to `to_process`
# for the next iteration.
while to_process:
    # Get the list of distinct values to process in this iteration.
    # Convert to list because we will modify the `to_process` set while adding new values generated.
    # Iterating over a copy (list) is safe while modifying the original set.
    current_values = list(to_process)
    # Clear the set for the next iteration's generated values.
    to_process.clear()

    # Process each distinct value from the current level
    # The order of processing within `current_values` does not affect the final total cost,
    # as multiplicities are handled by the `counts` dictionary.
    for x in current_values:
        # Check if this value `x` actually has any active occurrences to process.
        # `counts[x]` holds the total count of `x` accumulated from the previous level's expansions.
        # If `counts[x]` is 0, it means all occurrences of this value generated
        # in previous steps have already been processed (their cost added and replaced by children).
        if counts[x] > 0:
            # Get the total count of x accumulated from the previous level's expansions
            c = counts[x]

            # Add the cost for processing all `c` occurrences of `x`.
            # Each operation on `x` costs `x` yen. We perform this operation `c` times
            # (conceptually, processing all `c` instances of `x` that accumulated).
            total_cost += c * x

            # Mark these `c` occurrences of `x` as processed.
            # By setting counts[x] to 0, we ensure that the cost for *these* `c` occurrences
            # is added only once. Any new occurrences of `x` generated from processing other
            # numbers in `current_values` later in this same outer loop iteration will increase
            # `counts[x]` again, making it positive, and potentially adding `x` back to `to_process`
            # for the *next* outer loop iteration.
            counts[x] = 0

            # Calculate the two children values resulting from splitting x
            f = x // 2        # floor(x/2)
            ce = (x + 1) // 2 # ceil(x/2)

            # If the children values are >= 2, they need further processing in subsequent iterations.
            # The `c` occurrences of `x` are conceptually replaced by `c` occurrences of `f` and `c` occurrences of `ce`.
            # Add these counts to their respective entries in `counts`.
            # Add the child value to the `to_process` set for the next iteration if it's >= 2.
            if f >= 2:
                counts[f] += c
                # Add f to the set for the next iteration. Set handles duplicates, so adding
                # multiple times is fine and necessary if different parents generate the same f.
                to_process.add(f)

            if ce >= 2:
                counts[ce] += c
                # Add ce to the set for the next iteration.
                to_process.add(ce)

# The loop terminates when `to_process` is empty, meaning no numbers >= 2
# were generated in the previous step, indicating all numbers >= 2 have
# been processed and replaced by 1s.
# total_cost now holds the total amount paid throughout the entire process.

# Print the final answer
print(total_cost)