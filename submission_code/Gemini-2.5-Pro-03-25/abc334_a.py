import sys

def solve():
    # Read the two integers B and G from standard input
    line = sys.stdin.readline().split()
    b = int(line[0])
    g = int(line[1])

    # Compare the prices
    if b > g:
        # If the bat is more expensive
        print("Bat")
    else:
        # If the glove is more expensive (since B != G)
        print("Glove")

if __name__ == '__main__':
    solve()