def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    position = S.find("ABC")
    if position == -1:
        print(-1)
    else:
        print(position + 1)

# Call solve function
solve()