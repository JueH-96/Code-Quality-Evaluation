# YOUR CODE HERE
A, B = map(int, input().split())

# Two squares are horizontally adjacent if:
# 1. B - A = 1 (consecutive)
# 2. They don't cross row boundaries (B is not 4 or 7)
if B - A == 1 and B not in [4, 7]:
    print("Yes")
else:
    print("No")