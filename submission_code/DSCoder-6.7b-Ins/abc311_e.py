import sys

def read_input():
    H, W, N = map(int, sys.stdin.readline().split())
    holes = set(tuple(map(int, sys.stdin.readline().split())) for _ in range(N))
    return H, W, N, holes

def is_holeless(i, j, n, holes):
    for k in range(n):
        for l in range(n):
            if (i+k, j+l) in holes:
                return False
    return True

def count_holeless_squares(H, W, N, holes):
    count = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            for n in range(1, min(H-i+1, W-j+1)+1):
                if is_holeless(i, j, n, holes):
                    count += 1
    return count

H, W, N, holes = read_input()
print(count_holeless_squares(H, W, N, holes))