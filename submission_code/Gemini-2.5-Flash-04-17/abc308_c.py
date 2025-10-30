# YOUR CODE HERE
import sys
import functools

# Read N
N = int(sys.stdin.readline())

# Read A_i and B_i for each person and store with index
# Store as (A, B, 1-based index)
people_data = []
for i in range(N):
    A, B = map(int, sys.stdin.readline().split())
    people_data.append((A, B, i + 1))

# Define the comparison function for sorting
# We want to sort in descending order of success rate (A / (A+B))
# with ties broken by ascending order of index.
# Comparing the success rates A1/(A1+B1) and A2/(A2+B2):
# A1/(A1+B1) > A2/(A2+B2)
# Since A+B >= 1, the denominators A1+B1 and A2+B2 are positive.
# We can cross-multiply without changing the inequality direction:
# A1 * (A2+B2) > A2 * (A1+B1)
# Expand both sides:
# A1*A2 + A1*B2 > A2*A1 + A2*B1
# Since A1*A2 is equal to A2*A1, we can subtract it from both sides:
# A1*B2 > A2*B1
# So, rate1 > rate2 if and only if A1*B2 > A2*B1.

def compare_people(item1, item2):
    A1, B1, index1 = item1
    A2, B2, index2 = item2

    # Calculate the cross-multiplication difference: (A1*B2) - (A2*B1)
    # If positive, rate1 > rate2.
    # If negative, rate1 < rate2.
    # If zero, rate1 == rate2.
    # Use Python's arbitrary precision integers to handle large values of A and B.
    rate_comp_val = (A1 * B2) - (A2 * B1)

    if rate_comp_val > 0:
        # Rate 1 is greater than Rate 2, item1 should come before item2 (descending order)
        return -1
    elif rate_comp_val < 0:
        # Rate 1 is less than Rate 2, item1 should come after item2
        return 1
    else:
        # Rates are equal, break ties by ascending order of index
        if index1 < index2:
            # index1 is smaller, item1 comes before item2
            return -1
        elif index1 > index2:
            # index1 is larger, item1 comes after item2
            return 1
        else:
            # This case (indices are equal) should not happen for distinct people
            return 0

# Sort the people_data list using the custom comparison function
# functools.cmp_to_key adapts a cmp-style comparison function to a key function
sorted_people = sorted(people_data, key=functools.cmp_to_key(compare_people))

# Extract the 1-based indices from the sorted list
output_indices = [person[2] for person in sorted_people]

# Print the indices space-separated
print(*output_indices)