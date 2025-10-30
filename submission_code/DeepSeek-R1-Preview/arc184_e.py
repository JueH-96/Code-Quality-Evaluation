import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = []
    for _ in range(N):
        a = list(map(int, input[idx:idx+M]))
        idx += M
        A.append(a)
    
    from collections import defaultdict
    freq = defaultdict(int)
    for a in A:
        key = tuple(a)
        freq[key] += 1
    
    total = 0
    for cnt in freq.values():
        total += cnt * (cnt + 1) // 2
    print(total % MOD)

if __name__ == '__main__':
    main()