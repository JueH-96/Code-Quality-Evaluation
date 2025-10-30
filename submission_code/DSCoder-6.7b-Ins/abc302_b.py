import sys

def find_sequence(grid):
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    sequence = ['s', 'n', 'u', 'k', 'e']
    for h in range(H):
        for w in range(W-4):
            if S[h][w:w+5] == sequence:
                return [(h+1, w+1), (h+1, w+2), (h+1, w+3), (h+1, w+4), (h+1, w+5)]
    for w in range(W):
        for h in range(H-4):
            if sequence == [S[h+i][w] for i in range(5)]:
                return [(h+1, w+1), (h+2, w+1), (h+3, w+1), (h+4, w+1), (h+5, w+1)]
    for h in range(H-4):
        for w in range(W-4):
            if [S[h+i][w+i] for i in range(5)] == sequence:
                return [(h+1, w+1), (h+2, w+2), (h+3, w+3), (h+4, w+4), (h+5, w+5)]
    for h in range(H-4):
        for w in range(4, W):
            if [S[h+i][w-i] for i in range(5)] == sequence:
                return [(h+1, w+1), (h+2, w), (h+3, w-1), (h+4, w-2), (h+5, w-3)]
    return None

print(find_sequence(None))