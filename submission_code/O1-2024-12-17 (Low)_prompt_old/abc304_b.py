def solve():
    import sys
    N = int(sys.stdin.readline().strip())
    
    if N <= 999:
        print(N)
    elif N <= 9999:
        print((N // 10) * 10)
    elif N <= 99999:
        print((N // 100) * 100)
    elif N <= 999999:
        print((N // 1000) * 1000)
    elif N <= 9999999:
        print((N // 10000) * 10000)
    elif N <= 99999999:
        print((N // 100000) * 100000)
    else:  # up to 999999999
        print((N // 1000000) * 1000000)

def main():
    solve()

if __name__ == "__main__":
    main()