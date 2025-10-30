import sys

def solve():
    line = sys.stdin.readline().split()
    B = int(line[0])
    G = int(line[1])

    if B > G:
        print("Bat")
    else: # Since B != G is guaranteed, if B is not > G, then G must be > B
        print("Glove")

solve()