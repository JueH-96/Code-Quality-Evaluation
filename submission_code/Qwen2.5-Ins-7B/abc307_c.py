# YOUR CODE HERE
from itertools import product

def can AchieveGoal(HA, WA, A, HB, WB, B, HX, WX, X):
    def checkPaste(A, B, C):
        for dy, dx in product(range(HA + HB - HX + 1), range(WA + WB - WX + 1)):
            if all(C[i + dy][j + dx] == '#' or (A[i][j] == '#' or B[i][j] == '#') for i in range(HX) for j in range(WX)):
                return True
        return False

    C = [['.' for _ in range(WA + WB)] for _ in range(HA + HB)]
    for i in range(HA):
        for j in range(WA):
            C[i][j] = A[i][j]
    for i in range(HB):
        for j in range(WB):
            C[i + HA][j] = B[i][j]

    return checkPaste(A, B, C)

HA, WA = map(int, input().split())
A = [input() for _ in range(HA)]
HB, WB = map(int, input().split())
B = [input() for _ in range(HB)]
HX, WX = map(int, input().split())
X = [input() for _ in range(HX)]

if canAchieveGoal(HA, WA, A, HB, WB, B, HX, WX, X):
    print('Yes')
else:
    print('No')