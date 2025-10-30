def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, X, Y, Z = map(int, data)

    # Determine if traveling inbound or outbound
    if X < Y:
        # Inbound train goes from X up to Y
        if X <= Z <= Y:
            print("Yes")
        else:
            print("No")
    else:
        # Outbound train goes from X down to Y
        if Y <= Z <= X:
            print("Yes")
        else:
            print("No")

def main():
    solve()

if __name__ == "__main__":
    main()