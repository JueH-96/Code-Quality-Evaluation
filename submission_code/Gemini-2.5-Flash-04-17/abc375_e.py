import sys

# Set a large value for infinity
INF = float('inf')

def solve():
    # Read input
    N = int(sys.stdin.readline())
    people = []
    total_strength = 0
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        people.append((a, b))
        total_strength += b

    # Check if total strength is divisible by 3
    if total_strength % 3 != 0:
        print(-1)
        return

    # Calculate the target strength for each team
    target_strength = total_strength // 3

    # dp[s1][s2] will store the minimum switches required to achieve
    # a state where the sum of strengths in final team 1 is s1 and
    # the sum of strengths in final team 2 is s2, using a partition
    # of the people processed so far. The remaining strength will be in team 3.
    # We need to achieve s1 = target_strength and s2 = target_strength.
    # The maximum possible strength for s1 or s2 using a subset of people is total_strength.
    max_s = total_strength

    # Initialize DP table
    # dp[s1][s2] = minimum switches to get strength s1 in team 1 and s2 in team 2
    # by partitioning people processed so far.
    # Dimension: (max_s + 1) x (max_s + 1)
    dp = [[INF] * (max_s + 1) for _ in range(max_s + 1)]
    
    # Base case: Before processing any person, teams have 0 strength, 0 switches needed.
    dp[0][0] = 0 

    # Iterate through each person (initial_team, strength)
    for initial_team, strength in people:
        # Create a new DP table for the state after processing this person
        # This approach avoids using the result of the current person's assignment
        # when calculating the options for the same person.
        next_dp = [[INF] * (max_s + 1) for _ in range(max_s + 1)]

        # Iterate through all possible previous states (s1, s2)
        # These represent the strength sums for team 1 and team 2
        # using the people processed *before* the current one.
        for s1 in range(max_s + 1):
            for s2 in range(max_s + 1):
                if dp[s1][s2] == INF:
                    continue # Skip unreachable previous states

                # Option 1: Assign the current person to final team 1
                # The new sum for team 1 is s1 + strength. Team 2 sum remains s2.
                # Update next_dp[s1 + strength][s2].
                # The cost is the previous cost dp[s1][s2] plus 1 if the person switches.
                if s1 + strength <= max_s:
                    next_dp[s1 + strength][s2] = min(next_dp[s1 + strength][s2], dp[s1][s2] + (1 if initial_team != 1 else 0))

                # Option 2: Assign the current person to final team 2
                # The new sum for team 2 is s2 + strength. Team 1 sum remains s1.
                # Update next_dp[s1][s2 + strength].
                # The cost is the previous cost dp[s1][s2] plus 1 if the person switches.
                if s2 + strength <= max_s:
                    next_dp[s1][s2 + strength] = min(next_dp[s1][s2 + strength], dp[s1][s2] + (1 if initial_team != 2 else 0))

                # Option 3: Assign the current person to final team 3
                # The sums for team 1 and team 2 (s1, s2) do not change.
                # This person joins the group that forms team 3.
                # Update next_dp[s1][s2].
                # The cost is the previous cost dp[s1][s2] plus 1 if the person switches.
                next_dp[s1][s2] = min(next_dp[s1][s2], dp[s1][s2] + (1 if initial_team != 3 else 0))

        # The current next_dp becomes the dp table for the next person
        dp = next_dp

    # After processing all N people, dp[s1][s2] contains the minimum switches
    # to partition all N people such that team 1 sums to s1 and team 2 sums to s2.
    # The strength in team 3 is implicitly total_strength - s1 - s2.
    # For the teams to have equal strength, we need s1 = target_strength and s2 = target_strength.
    # In this case, strength in team 3 will be total_strength - target_strength - target_strength = 3*target_strength - 2*target_strength = target_strength.
    result = dp[target_strength][target_strength]

    # If the state (target_strength, target_strength) is unreachable, it's impossible to balance teams
    if result == INF:
        print(-1)
    else:
        print(result)

# Run the solve function
solve()