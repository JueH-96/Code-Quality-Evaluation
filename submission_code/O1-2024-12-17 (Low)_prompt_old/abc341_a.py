def solve():
    import sys
    data = sys.stdin.read().strip()
    N = int(data)
    result = '10' * N + '1'
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()