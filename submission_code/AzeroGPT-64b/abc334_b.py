A, M, L, R = map(int, input().split())

def check(x):
    # Check if x is to the right of (or at) the closest Christmas tree to the left of L
    left = (A - x) % M <= 0
    # Check if x is to the left of (or at) the closest Christmas tree to the right of R
    right = (A - x) % M >= 0
    
    return (L <= x and left) or (x <= R and right)

left = A - (A - L) % M if A <= L else A - (A - L) % M + M
right = A - (A - R) % M if A >= R else A - (A - R) % M + M

if check(left) and check(right):
    print(max(0, (right - left) // M + 1))
else:
    print(0)