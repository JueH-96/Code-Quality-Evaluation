import sys

def can_capture(i, j, a, b):
    return i == a or j == b or i + j == a + b or i - j == a - b

def solve():
    N, M = map(int, input().split())
    pieces = []
    for _ in range(M):
        a, b = map(int, input().split())
        pieces.append((a, b))
    
    count = 0
    for i in range(1, N+1):
        for j in range(1, N+1):
            can_place = True
            for a, b in pieces:
                if can_capture(i, j, a, b):
                    can_place = False
                    break
            if can_place:
                count += 1
    
    return count

print(solve())