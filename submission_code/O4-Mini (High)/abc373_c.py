import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # Next n integers are A, following n integers are B
    # We only need the maxima of each sequence.
    a_max = -10**18
    b_max = -10**18
    idx = 1
    for _ in range(n):
        val = int(data[idx])
        if val > a_max:
            a_max = val
        idx += 1
    for _ in range(n):
        val = int(data[idx])
        if val > b_max:
            b_max = val
        idx += 1
    print(a_max + b_max)

if __name__ == "__main__":
    main()