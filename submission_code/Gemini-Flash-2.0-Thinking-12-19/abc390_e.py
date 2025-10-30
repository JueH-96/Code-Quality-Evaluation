import sys

def solve():
    # Read input N and X
    N, X = map(int, sys.stdin.readline().split())

    # Read food details
    foods = []
    for _ in range(N):
        v, a, c = map(int, sys.stdin.readline().split())
        foods.append((v, a, c))

    # dp state: dp[(v1, v2, v3)] = minimum calories to achieve exactly v1, v2, v3 vitamin units.
    # We use a dictionary (map) to store only reachable states with finite calories
    # that do not exceed the total calorie limit X.
    # The keys are tuples representing (total vitamin 1, total vitamin 2, total vitamin 3).
    # The values are the minimum calories required to reach that exact combination of vitamins.
    dp = {(0, 0, 0): 0}

    # Iterate through each food item
    for v, a, c in foods:
        # Create a new dictionary for the next step, initially copying the current states.
        # This is necessary because we are considering *either* taking the current food
        # or not taking it. The states achievable without the current food are already in dp.
        # We now add the states achievable by adding the current food to existing states.
        next_dp = dp.copy()

        # Iterate through all states currently in our DP table
        for (v1, v2, v3), current_c in dp.items():
            # Calculate the calorie cost if we add the current food
            new_c = current_c + c

            # If the new calorie cost exceeds the limit X, this state is not reachable
            # within the budget, so we ignore it.
            if new_c > X:
                continue

            # Calculate the new vitamin amounts if we add the current food
            nv1, nv2, nv3 = v1, v2, v3
            if v == 1:
                nv1 += a
            elif v == 2:
                nv2 += a
            elif v == 3:
                nv3 += a

            new_state = (nv1, nv2, nv3)

            # If this new state (vitamin combination) is not yet recorded in next_dp,
            # or if the new calorie cost is lower than the previously recorded cost
            # for this vitamin combination, update or record the new state with the lower cost.
            if new_state not in next_dp or new_c < next_dp[new_state]:
                next_dp[new_state] = new_c

        # After considering the current food in combination with all previous states,
        # the next step's DP table is complete.
        dp = next_dp

    # After iterating through all foods, the dp dictionary contains all reachable
    # vitamin combinations with their minimum calorie costs (not exceeding X).

    max_min_vitamin = 0
    # Iterate through all reachable states (vitamin combinations) in the final DP table
    # and find the maximum value among the minimum of the three vitamin amounts.
    for (v1, v2, v3), cost in dp.items():
        max_min_vitamin = max(max_min_vitamin, min(v1, v2, v3))

    # Print the maximum possible value of the minimum intake among vitamins 1, 2, and 3.
    print(max_min_vitamin)

solve()