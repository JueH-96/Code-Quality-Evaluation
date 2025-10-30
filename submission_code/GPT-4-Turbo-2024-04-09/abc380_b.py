import sys
input = sys.stdin.read

def solve():
    S = input().strip()
    counts = []
    count = 0
    for char in S:
        if char == '|':
            if count > 0:
                counts.append(count)
            count = 0
        elif char == '-':
            count += 1
    print(" ".join(map(str, counts)))

solve()