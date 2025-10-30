import sys

def solve():
    # Read N and D
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    D = int(line1[1])

    # Read the initial state string S
    S = sys.stdin.readline().strip()

    # Convert the string to a list of characters for mutability
    boxes = list(S)

    # Simulate D days of eating cookies
    for _ in range(D):
        # Find the index of the rightmost cookie ('@')
        rightmost_index = -1
        # Iterate from right to left
        for i in range(N - 1, -1, -1):
            if boxes[i] == '@':
                rightmost_index = i
                break # Found the rightmost cookie, stop searching

        # Eat the cookie (change '@' to '.')
        # The problem guarantees that S contains at least D '@'s,
        # so rightmost_index will always be found (>= 0).
        boxes[rightmost_index] = '.'

    # Convert the list back to a string and print the final state
    print("".join(boxes))

# Execute the solve function
solve()