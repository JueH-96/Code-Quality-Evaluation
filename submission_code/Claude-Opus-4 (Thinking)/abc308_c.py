n = int(input())
people = []
for i in range(n):
    a, b = map(int, input().split())
    people.append((i + 1, a, b))

def compare(p1, p2):
    num1, a1, b1 = p1
    num2, a2, b2 = p2
    # Compare a1/(a1+b1) vs a2/(a2+b2)
    # Cross multiply: a1*(a2+b2) vs a2*(a1+b1)
    left = a1 * (a2 + b2)
    right = a2 * (a1 + b1)
    if left > right:
        return -1  # p1 has higher success rate, comes first
    elif left < right:
        return 1   # p2 has higher success rate, comes first
    else:
        # Same success rate, sort by person number ascending
        return num1 - num2

from functools import cmp_to_key
people.sort(key=cmp_to_key(compare))

print(' '.join(str(p[0]) for p in people))