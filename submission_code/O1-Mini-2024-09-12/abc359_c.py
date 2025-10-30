# YOUR CODE HERE
import sys

def solve():
    S_x, S_y = map(int, sys.stdin.readline().split())
    T_x, T_y = map(int, sys.stdin.readline().split())
    dy = abs(T_y - S_y)
    print(dy)

if __name__ == "__main__":
    solve()