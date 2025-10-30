def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    if N < 1000:
        # N is 0 through 999, print as is.
        print(N)
    elif N < 10000:
        # N is between 1000 and 9999: truncate the ones digit.
        print((N // 10) * 10)
    elif N < 100000:
        # N is between 10000 and 99999: truncate the tens digit and all digits below.
        print((N // 100) * 100)
    elif N < 1000000:
        # N is between 100000 and 999999: truncate the hundreds digit and all digits below.
        print((N // 1000) * 1000)
    elif N < 10000000:
        # N is between 1000000 and 9999999: truncate the thousands digit and all digits below.
        print((N // 10000) * 10000)
    elif N < 100000000:
        # N is between 10000000 and 99999999: truncate the ten-thousands digit and all digits below.
        print((N // 100000) * 100000)
    else:
        # N is between 100000000 and 999999999: truncate the hundred-thousands digit and all digits below.
        print((N // 1000000) * 1000000)

if __name__ == '__main__':
    main()