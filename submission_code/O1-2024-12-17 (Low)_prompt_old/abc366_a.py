def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, A = map(int, data)

    # Check if Takahashi or Aoki has already more than half of the votes (since N is odd)
    if 2 * T > N:
        print("Yes")
    elif 2 * A > N:
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()