import sys

def can_be_expressed(N):
    if N < 1:
        return False
    original_N = N
    for prime in [2, 3]:
        while N % prime == 0:
            N //= prime
    return N == 1

def main():
    input = sys.stdin.read().strip()
    if not input:
        print("No")
        return
    try:
        N = int(input)
    except ValueError:
        print("No")
        return
    if can_be_expressed(N):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()