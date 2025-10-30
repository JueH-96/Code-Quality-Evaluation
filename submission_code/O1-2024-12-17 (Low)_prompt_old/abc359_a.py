def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    strings = data[1:]
    count_takahashi = sum(1 for s in strings if s == "Takahashi")
    print(count_takahashi)

# Call the solve function
if __name__ == "__main__":
    solve()