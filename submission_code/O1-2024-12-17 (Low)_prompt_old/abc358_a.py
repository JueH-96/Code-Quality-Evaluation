def solve():
    import sys

    data = sys.stdin.read().strip().split()
    S, T = data[0], data[1]
    
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")

# Call the solve function
solve()