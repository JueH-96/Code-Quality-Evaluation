MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    initial = [i % 2 for i in range(1, N+1)]
    
    # Check if the target is possible
    possible = True
    res = 1
    i = 0
    while i < N:
        if A[i] == 1:
            # Find the end of the 1's block
            j = i
            while j < N and A[j] == 1:
                j += 1
            # Check if the block starts and ends with 1's in initial
            if initial[i] != 1 or initial[j-1] != 1:
                possible = False
                break
            # Count zeros in initial within this block
            zeros = 0
            for k in range(i, j):
                if initial[k] == 0:
                    zeros += 1
            # Multiply by (zeros + 1)
            res = res * (zeros + 1) % MOD
            i = j
        else:
            # Check all are 0 in initial
            for k in range(i, j):
                if initial[k] != 0:
                    possible = False
                    break
            if not possible:
                break
            i = j
    
    if possible:
        print(res)
    else:
        print(0)

if __name__ == '__main__':
    main()