A, B = map(int, input().split())

if A == B:
    print(-1)
else:
    # The culprit is the one not mentioned by either witness
    # Since A and B are different and each is one of 1, 2, or 3, the culprit is the missing number
    culprit = 6 - A - B  # Since 1 + 2 + 3 = 6, the missing one is 6 - A - B
    print(culprit)