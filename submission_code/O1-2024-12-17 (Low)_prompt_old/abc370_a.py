def solve():
    import sys

    data = sys.stdin.read().strip().split()
    L, R = map(int, data)

    # If both hands or no hand are raised
    if (L == 0 and R == 0) or (L == 1 and R == 1):
        print("Invalid")
    else:
        # If only left hand is raised
        if L == 1:
            print("Yes")
        else:
            # Only right hand is raised
            print("No")

def main():
    solve()

if __name__ == "__main__":
    main()