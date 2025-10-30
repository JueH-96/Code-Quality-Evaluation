memo = {1: 0}

def total_payment(N):
    if N in memo:
        return memo[N]
    floor = N // 2
    ceil = (N + 1) // 2
    result = N + total_payment(floor) + total_payment(ceil)
    memo[N] = result
    return result

def main():
    import sys
    N = int(sys.stdin.read())
    print(total_payment(N))

if __name__ == "__main__":
    main()