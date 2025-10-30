import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    a = []
    if N >= 2:
        a = [0] * (N - 1)
        for i in range(N - 1):
            if S[i] == S[i + 1]:
                a[i] = 1

    # Compute prefix sums
    prefix = [0] * (len(a) + 1)
    for i in range(len(a)):
        prefix[i + 1] = prefix[i] + a[i]

    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        if l >= r:
            print(0)
            continue
        x = l - 1
        y = r - 2
        if x > y:
            print(0)
        else:
            res = prefix[y + 1] - prefix[x]
            print(res)

if __name__ == "__main__":
    main()