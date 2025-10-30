def solve():
    import sys

    data = sys.stdin.read().strip()
    N = int(data)

    # Keep dividing N by 2 while it's divisible by 2
    while N % 2 == 0:
        N //= 2

    # Keep dividing N by 3 while it's divisible by 3
    while N % 3 == 0:
        N //= 3

    # If after removing all factors of 2 and 3, we are left with 1, 
    # then N can be represented as 2^x3^y.
    if N == 1:
        print("Yes")
    else:
        print("No")

def main():
    solve()

if __name__ == "__main__":
    main()