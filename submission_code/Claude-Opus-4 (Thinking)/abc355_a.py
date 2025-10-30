A, B = map(int, input().split())

if A == B:
    print(-1)
else:
    # The culprit is the person who is neither A nor B
    # Since there are only 3 people (1, 2, 3), we can find the culprit
    # by summing 1+2+3=6 and subtracting A and B
    culprit = 6 - A - B
    print(culprit)