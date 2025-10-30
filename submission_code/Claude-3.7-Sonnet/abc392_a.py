# Read the three integers from the input
a, b, c = map(int, input().split())

# Check if any permutation satisfies the condition B_1 * B_2 = B_3
if a * b == c or a * c == b or b * c == a:
    print("Yes")
else:
    print("No")