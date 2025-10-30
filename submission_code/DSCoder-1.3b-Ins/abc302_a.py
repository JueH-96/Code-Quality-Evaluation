import sys

def calculate_attacks(A, B):
    attacks = 0
    while A > 0:
        A -= B
        attacks += 1
    return attacks

for line in sys.stdin:
    A, B = map(int, line.split())
    print(calculate_attacks(A, B))