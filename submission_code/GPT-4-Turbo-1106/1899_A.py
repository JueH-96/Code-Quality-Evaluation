def who_wins(n):
    # Vanya wins if the number is already divisible by 3
    if n % 3 == 0:
        return "First"
    # Otherwise, Vanya will try to make the number divisible by 3 in his first move
    # If n is 1 mod 3, Vanya will subtract 1 to make it divisible by 3
    # If n is 2 mod 3, Vanya will add 1 to make it divisible by 3
    # In both cases, Vanya wins in the first move
    else:
        return "First"

# Read the number of test cases
t = int(input().strip())

# Iterate over each test case
for _ in range(t):
    # Read the integer n
    n = int(input().strip())
    # Print the winner
    print(who_wins(n))