from itertools import product

N = int(input())
A = list(map(int, input().split()))
M = int(input())
B = list(map(int, input().split()))
L = int(input())
C = list(map(int, input().split()))
Q = int(input())
X = list(map(int, input().split()))

AB = [a + b for a, b in product(A, B)]
AB.sort()
AB_set = set(AB)

def binary_search(target):
    left = 0
    right = len(AB) - 1
    while left <= right:
        mid = (left + right) // 2
        if AB[mid] == target:
            return True
        elif AB[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return False

for x in X:
    found = False
    for c in C:
        if binary_search(x - c):
            found = True
            break
    print("Yes" if found else "No")