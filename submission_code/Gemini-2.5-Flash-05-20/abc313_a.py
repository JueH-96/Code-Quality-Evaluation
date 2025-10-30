import sys

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # P_1 is the programming ability of person 1
    P_1 = P[0]

    # If there is only one person, person 1 is already the strongest
    # and there are no other people to compare against.
    # So, 0 additional points are needed.
    if N == 1:
        print(0)
        return

    # Find the maximum programming ability among all other people (i.e., P_2 to P_N)
    # We use slicing P[1:] to get the abilities of people from 2 to N.
    P_max_others = max(P[1:])

    # To become strictly stronger than P_max_others, person 1's score (P_1 + x)
    # must be at least P_max_others + 1.
    # So, we need P_1 + x >= P_max_others + 1
    # This implies x >= (P_max_others + 1) - P_1
    # The minimum integer x that satisfies this is (P_max_others + 1) - P_1.
    
    # Calculate the target score person 1 needs to reach
    target_score_for_P1 = P_max_others + 1

    # Calculate the points needed: difference between target score and current score
    points_needed = target_score_for_P1 - P_1

    # The problem asks for a non-negative integer x.
    # If P_1 is already greater than P_max_others, or exactly P_max_others + 1,
    # then points_needed might be 0 or negative. In such cases, 0 points are needed.
    # We take the maximum of 0 and the calculated points_needed.
    result = max(0, points_needed)

    print(result)

# Call the solve function to execute the logic
solve()