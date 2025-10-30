# YOUR CODE HERE
import sys
sys.setrecursionlimit(1 << 25)
MOD = 998244353

def main():
    import sys
    input = sys.stdin.readline
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))
    N = len(A_list)
    lengths = []
    pow_10_len = []
    for num in A_list:
        l = len(str(num))
        lengths.append(l)
    pow_10_len = [pow(10, l, MOD) for l in lengths]
    prefix_sum = 0
    total = 0
    for idx, (A_j, len_j, pow_10_len_j) in enumerate(zip(A_list, lengths, pow_10_len), start=1):
        if idx >= 2:
            total += prefix_sum * pow_10_len_j
            total %= MOD
        total += A_j * (idx -1)
        total %= MOD
        prefix_sum += A_j
        prefix_sum %= MOD
    print(total)

if __name__ == "__main__":
    main()