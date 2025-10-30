# YOUR CODE HERE
import sys

def solve():
    # Read the number of dishes
    n = int(sys.stdin.readline())

    # Read the sequence of dishes into a list
    # .strip() removes potential trailing newline characters
    dishes = [sys.stdin.readline().strip() for _ in range(n)]

    # Assume Takahashi can eat all dishes initially.
    # We will set this to False if we find a condition that prevents him.
    can_eat_all = True

    # Iterate through the dishes to check for consecutive sweet ones.
    # We need to look at pairs of adjacent dishes: (dishes[i], dishes[i+1]).
    # The loop index 'i' should go from 0 up to n-2 (the second-to-last dish).
    # range(n - 1) generates indices 0, 1, ..., n-2.
    for i in range(n - 1):
        # Check if the dish at index 'i' and the dish at index 'i+1' are both "sweet"
        if dishes[i] == "sweet" and dishes[i+1] == "sweet":
            # If two consecutive sweet dishes are found, Takahashi feels sick
            # *after* eating the second one (the dish at index i+1).
            # He is then unable to eat any *more* dishes (i.e., dish i+2 and onwards).

            # The question is whether he can eat *all* N dishes.
            # He fails to eat all N dishes only if he gets sick *before* having eaten the last dish.
            # The last dish is at index n-1.
            # He gets sick after eating the dish at index i+1.
            # If this sickness-inducing dish (at index i+1) is *before* the last dish (index n-1),
            # meaning i+1 < n-1, then he will not be able to eat the subsequent dishes,
            # including the last one. In this case, he cannot eat all N dishes.
            if i + 1 < n - 1:
                # Set the flag to False indicating he cannot eat all dishes.
                can_eat_all = False
                # Since we have found a condition that prevents him from eating all dishes,
                # we can stop checking further pairs.
                break

    # After the loop finishes (either by checking all pairs or breaking early),
    # print the result based on the final state of the 'can_eat_all' flag.
    if can_eat_all:
        # If the flag is still True, it means either:
        # 1. No consecutive sweet dishes were found at all.
        # 2. Consecutive sweet dishes were found only as the very last pair
        #    (dishes at indices n-2 and n-1). In this case, he still managed
        #    to eat all N dishes before feeling sick.
        print("Yes")
    else:
        # If the flag is False, it means consecutive sweet dishes were found
        # earlier in the sequence, preventing him from finishing all N dishes.
        print("No")

# Execute the solution function
solve()