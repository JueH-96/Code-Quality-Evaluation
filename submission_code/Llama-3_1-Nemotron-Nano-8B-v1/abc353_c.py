import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    MOD = 10**8
    B = [a % MOD for a in A]
    
    sorted_list = []
    K = 0
    for i in range(N-1, -1, -1):
        b = B[i]
        target = MOD - b
        pos = bisect.bisect_left(sorted_list, target)
        K += len(sorted_list) - pos
        bisect.insort(sorted_list, b)
    
    sum_B = sum(B)
    total = sum_B * (N-1) - MOD * K
    print(total)

if __name__ == '__main__':
    main()