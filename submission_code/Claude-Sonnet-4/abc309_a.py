# YOUR CODE HERE
A, B = map(int, input().split())

# Define horizontally adjacent pairs on a 3x3 board (1-9)
# Board layout:
# 1 2 3
# 4 5 6
# 7 8 9

horizontal_pairs = [(1,2), (2,3), (4,5), (5,6), (7,8), (8,9)]

if (A, B) in horizontal_pairs:
    print("Yes")
else:
    print("No")