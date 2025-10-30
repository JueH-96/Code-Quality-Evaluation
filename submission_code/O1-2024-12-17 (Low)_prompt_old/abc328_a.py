def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = int(data[1])
    scores = list(map(int, data[2:]))

    total = sum(s for s in scores if s <= X)
    print(total)

# Call the solve function
solve()