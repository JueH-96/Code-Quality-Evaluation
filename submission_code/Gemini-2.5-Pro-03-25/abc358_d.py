import sys

# Function to read input and solve the problem
def solve():
    # Read N (number of boxes) and M (number of people)
    # n: number of available boxes
    # m: number of people requiring boxes
    n, m = map(int, sys.stdin.readline().split())

    # Read the list of box prices/candies A
    # A_i is both the price and the number of candies in box i.
    a = list(map(int, sys.stdin.readline().split()))

    # Read the list of minimum required candies B for each person
    # B_i is the minimum number of candies person i needs.
    b = list(map(int, sys.stdin.readline().split()))

    # Constraint check: If the number of people (M) is greater than the number of available boxes (N),
    # it's impossible to give each person a distinct box.
    # The problem constraints state M <= N, but this check is good practice.
    if m > n:
        print("-1")
        return

    # Sort the available boxes by price/candy count in ascending order.
    # This allows us to consider cheaper boxes first when trying to satisfy requirements.
    a.sort()

    # Sort the minimum requirements B in ascending order.
    # Processing requirements from the smallest to the largest allows a greedy strategy.
    # We try to satisfy the smallest requirement with the cheapest possible box,
    # then the next smallest requirement with the next cheapest possible box, and so on.
    # This greedy approach works because if we use a more expensive box than necessary
    # for a smaller requirement, that box might be needed for a larger requirement later,
    # potentially leading to a higher total cost or making it impossible to satisfy all requirements.
    b.sort()

    total_cost = 0 # Initialize the total cost of the selected boxes.
    a_ptr = 0 # Pointer to the current box being considered in the sorted list 'a'.
              # It represents the index of the first available box not yet assigned.
    possible = True # Flag to track if a valid assignment satisfying all requirements has been found.

    # Iterate through each person's requirement (in sorted order B_1, B_2, ..., B_M).
    for i in range(m):
        required_candies = b[i] # The minimum number of candies needed for the current person (person i in the sorted requirement list).

        # Find the first available box (starting from index a_ptr) in the sorted list 'a'
        # that has at least 'required_candies'.
        # We advance 'a_ptr' past any boxes that are too small for the current requirement.
        # The `while` loop efficiently finds the smallest index `j >= a_ptr` such that `a[j] >= required_candies`.
        while a_ptr < n and a[a_ptr] < required_candies:
            a_ptr += 1

        # Check if we have exhausted the available boxes ('a') before finding
        # a suitable one for the current requirement.
        if a_ptr == n:
            # If a_ptr reached the end of list 'a' (index n), it means no suitable box
            # (with >= required_candies) was found among the remaining available boxes.
            # Therefore, it's impossible to satisfy this requirement and, consequently, all requirements.
            possible = False
            break # Exit the loop early, as we know a solution is impossible.

        # If a suitable box is found (i.e., a_ptr < n):
        # The box `a[a_ptr]` satisfies the current requirement (`a[a_ptr] >= required_candies`).
        # Since we sorted 'a' and advanced 'a_ptr' sequentially, `a[a_ptr]` is the
        # cheapest available box that meets this requirement.
        total_cost += a[a_ptr] # Add the cost (which is equal to the candy count a[a_ptr]) of this box to the total cost.
        
        # Move the pointer to the next box. This effectively "assigns" or "uses" the box a[a_ptr]
        # for the current requirement and ensures that the search for the next requirement
        # starts from the subsequent box (index a_ptr+1). This prevents using the same box twice.
        a_ptr += 1 

    # After iterating through all M requirements (or breaking early if impossible):
    if possible:
        # If the loop completed successfully without breaking (possible is still True),
        # it means we found suitable boxes for all M requirements using the greedy strategy.
        # The calculated total_cost represents the minimum possible cost.
        print(total_cost)
    else:
        # If the loop was broken because a requirement could not be met (possible is False),
        # print -1 to indicate that it's impossible to satisfy the condition.
        print("-1")

# Execute the solve function when the script runs.
# This reads input from stdin, processes it, and prints the output to stdout.
solve()