LIMIT = 10**9

def main():
    import sys
    n, m = map(int, sys.stdin.readline().split())
    total = 0
    current = 1
    for i in range(m + 1):
        total += current
        if total > LIMIT:
            print("inf")
            return
        if i < m:
            current *= n
    print(total)

if __name__ == "__main__":
    main()