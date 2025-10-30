import math

def find_spoiled_bottle(n):
    # Calculate the minimum number of friends needed
    m = math.ceil(math.log2(n))

    # Print the number of friends
    print(m)

    # Create a list to store the bottles for each friend
    bottles = [[] for _ in range(m)]

    # Distribute the bottles to the friends
    for i in range(n):
        for j in range(m):
            if (i >> j) & 1:
                bottles[j].append(i + 1)

    # Print the bottles for each friend
    for i in range(m):
        print(len(bottles[i]), *bottles[i])

    # Get the result from the judge
    result = input()

    # Find the spoiled bottle
    spoiled_bottle = 0
    for i in range(m):
        if result[i] == '1':
            spoiled_bottle |= 1 << i

    # Print the spoiled bottle
    print(spoiled_bottle + 1)

# Read the input
n = int(input())

# Find the spoiled bottle
find_spoiled_bottle(n)