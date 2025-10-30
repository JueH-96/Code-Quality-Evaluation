# YOUR CODE HERE
A, B, C = map(int, input().split())
total = A + B + C
if total % 2 == 0:
    half = total // 2
    if A == half or B == half or C == half or (A + B) == half or (A + C) == half or (B + C) == half:
        print("Yes")
    else:
        print("No")
else:
    print("No")