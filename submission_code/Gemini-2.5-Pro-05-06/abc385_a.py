# Read the three integers A, B, C from standard input
a, b, c = map(int, input().split())

# Condition for forming two groups with equal sums:
# One number is equal to the sum of the other two.
# e.g., groups {a} and {b,c}. Sums are 'a' and 'b+c'. Equal if a = b+c.
cond_two_groups = (a == b + c) or \
                  (b == a + c) or \
                  (c == a + b)

# Condition for forming three groups with equal sums:
# All three numbers are equal.
# Groups {a}, {b}, {c}. Sums are 'a', 'b', 'c'. Equal if a = b = c.
cond_three_groups = (a == b and b == c)

# If either condition is met, it's possible.
if cond_two_groups or cond_three_groups:
    print("Yes")
else:
    print("No")