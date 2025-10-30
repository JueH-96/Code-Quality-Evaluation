import sys
input = sys.stdin.readline

def main():
    N = int(input())
    import math
    M = math.ceil(math.log2(N))
    print(M)
    for i in range(M):
        binary = bin(i)[2:].zfill(M)
        K = sum([1 if binary[j]=='1' else 0 for j in range(M)])
        A = ' '.join(str(M - j) if binary[j] == '1' else '' for j in range(M)).strip()
        if A == '':
            print('0')
        else:
            print(A)
        sys.stdout.flush()


    S = input().strip()

    if '1' not in S:
        print(1)
        sys.stdout.flush()
        return

    if '0' not in S:
        print(N)
        sys.stdout.flush()
        return

    S = [int(s) for s in S]

    X = 0
    for i, n in enumerate(S):
        if n == 1:
            X |= (1 << (M - i - 1))

    print(X + 1)
    sys.stdout.flush()
    return


main()