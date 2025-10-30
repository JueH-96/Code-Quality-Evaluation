import math

def truncate(N, factor):
    return (N // factor) * factor

def main():
    N = int(input())
    if N == 0:
        print(0)
        return
    k = len(str(N))  # Number of digits
    if k <= 3:
        print(N)
    else:
        if k == 4:
            factor = 10
        elif k == 5:
            factor = 100
        elif k == 6:
            factor = 1000
        elif k == 7:
            factor = 1000
        elif k == 8:
            factor = 100000
        elif k == 9:
            factor = 100000
        else:
            factor = 1  # For any unexpected cases, though N is bounded
        truncated = truncate(N, factor)
        print(truncated)

if __name__ == "__main__":
    main()