# Read the number of people
N = int(input())

# Read the list of scores
P = list(map(int, input().split()))

# Get the score of person 1
P1_score = P[0]

# If N is 1, person 1 is the only person and thus the strongest, needs 0 points.
# The condition "P_1 + x > P_i for all i != 1" is vacuously true when N=1 as there are no i != 1.
# The minimum non-negative x is 0.
if N == 1:
    print(0)
else:
    # Get the scores of other people (all elements except the first one)
    P_others = P[1:]

    # Find the maximum score among the other people.
    # This list P_others is guaranteed to be non-empty because N > 1.
    P_max_others = max(P_others)

    # To be strictly stronger than everyone else, person 1's score
    # must be at least (maximum score among others) + 1.
    P_target = P_max_others + 1

    # Calculate how many more points person 1 needs to reach the target score.
    points_needed = P_target - P1_score

    # The answer is the minimum non-negative number of points needed.
    # If points_needed is positive or zero, that's the answer.
    # If points_needed is negative, it means P1_score is already >= P_target,
    # i.e., P1_score > P_max_others. In this case, person 1 is already
    # strictly stronger than others, and needs 0 points.
    print(max(0, points_needed))