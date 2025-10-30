def solve():
    import sys
    data = sys.stdin.read().strip()
    N = int(data)

    # Find all divisors j of N in the range 1..9
    divisors = [j for j in range(1, 10) if N % j == 0]

    result = []
    for i in range(N + 1):
        # Find the smallest j such that i is a multiple of N//j
        chosen = None
        for j in divisors:
            if i % (N // j) == 0:
                chosen = j
                break
        # If found such j, use its digit
        result.append(str(chosen) if chosen else '-')

    print("".join(result))

def main():
    solve()

if __name__ == "__main__":
    main()