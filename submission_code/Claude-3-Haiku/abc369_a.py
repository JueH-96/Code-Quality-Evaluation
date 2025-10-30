import sys

def count_arithmetic_sequences(A, B):
    count = 0
    for x in range(1, 101):
        if (A - x) * (B - x) <= 0 and (A - x) % (B - x) == 0:
            count += 1
    return count

A, B = map(int, sys.stdin.readline().split())
print(count_arithmetic_sequences(A, B))