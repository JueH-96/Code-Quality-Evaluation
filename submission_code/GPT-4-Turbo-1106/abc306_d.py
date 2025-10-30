def max_tastiness(N, courses):
    # Initialize variables
    max_tasty = 0
    current_tasty = 0
    min_antidote = float('inf')
    has_poison = False

    # Iterate through the courses
    for X, Y in courses:
        if X == 0:  # Antidotal course
            # If it's an antidotal course, we can always eat it
            current_tasty += Y
            # Keep track of the smallest antidote tastiness
            min_antidote = min(min_antidote, Y)
        else:  # Poisonous course
            # We can only eat a poisonous course if we have eaten an antidote before
            if current_tasty > 0:
                current_tasty += Y
                has_poison = True
            # If we haven't eaten anything yet, we skip the poisonous course

    # If we have eaten a poison, we need to subtract the smallest antidote
    # to ensure we don't die (if we have only eaten poisons, we don't do anything)
    if has_poison and min_antidote != float('inf'):
        current_tasty -= min_antidote

    # The maximum tastiness is the max of the current tastiness and 0
    max_tasty = max(current_tasty, max_tasty)

    return max_tasty

# Read input
N = int(input().strip())
courses = [tuple(map(int, input().split())) for _ in range(N)]

# Calculate and print the result
print(max_tastiness(N, courses))