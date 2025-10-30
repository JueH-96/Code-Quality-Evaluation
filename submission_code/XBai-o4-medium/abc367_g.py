import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    
    MOD = 998244353
    
    # Initialize cnt: list of M dicts
    cnt = [{} for _ in range(M)]
    cnt[0] = {0: 1}  # empty subset
    
    for a in A:
        # collect all (j, x, c) in current cnt
        old_entries = []
        for j in range(M):
            for x, c in cnt[j].items():
                old_entries.append((j, x, c))
        # process each entry
        for j, x, c in old_entries:
            new_j = (j + 1) % M
            new_x = x ^ a
            if new_x in cnt[new_j]:
                cnt[new_j][new_x] += c
            else:
                cnt[new_j][new_x] = c
    
    # Now calculate the answer: sum x^k * count[x] for x in cnt[0], minus 0^k (which is zero)
    res = 0
    for x, c in cnt[0].items():
        res = (res + pow(x, K, MOD) * c) % MOD
    
    print(res % MOD)

if __name__ == "__main__":
    main()