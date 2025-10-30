def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # We want to check if 'ab' or 'ba' is a substring of S
    # A simple way: just check all adjacent pairs
    for i in range(N - 1):
        if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
            print("Yes")
            return
    print("No")

# Let's call the solve() function:
solve()