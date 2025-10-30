import sys

def approximate_number(N):
    if N <= 999:
        return N
    elif N <= 9999:
        return N // 10 * 10
    elif N <= 99999:
        return N // 100 * 100
    elif N <= 999999:
        return N // 1000 * 1000
    elif N <= 9999999:
        return N // 10000 * 10000
    elif N <= 99999999:
        return N // 100000 * 100000
    elif N <= 999999999:
        return N // 1000000 * 1000000

def main():
    input = sys.stdin.read()
    N = int(input.strip())
    result = approximate_number(N)
    print(result)

if __name__ == "__main__":
    main()