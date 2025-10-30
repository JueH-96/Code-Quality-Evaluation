def solve():
    # Read the input values for A and B
    A, B = map(int, input().split())

    # Case 1: A and B are the same person
    if A == B:
        # Both witnesses say the same person is not the culprit.
        # This means one person is excluded, leaving two possible culprits.
        # The culprit cannot be uniquely identified.
        print(-1)
    # Case 2: A and B are different people
    else:
        # The witnesses say two different people are not the culprits.
        # The suspects are 1, 2, 3.
        # If A and B are excluded, there is only one person remaining.
        # The sum of the suspect numbers is 1 + 2 + 3 = 6.
        # The number of the remaining person (the culprit) is 6 - A - B.
        culprit = 6 - A - B
        print(culprit)

if __name__ == '__main__':
    solve()