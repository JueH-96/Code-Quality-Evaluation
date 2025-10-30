def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Check if all values in A are the same
    if len(set(A)) == 1:
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()