from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    # Compute prefix sums
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i-1] + A[i-1]
    
    # Compute mods
    mods = [0] * (N + 1)
    for i in range(N + 1):
        mods[i] = prefix[i] % M
    
    # Case 1: s < t
    freq = defaultdict(int)
    freq[mods[0]] = 1  # s=1's mod is mods[0]
    total_case1 = 0
    for t in range(2, N + 1):
        mod_t = mods[t-1]
        total_case1 += freq.get(mod_t, 0)
        freq[mod_t] += 1
    
    # Case 2: s > t
    freq2 = defaultdict(int)
    total_case2 = 0
    for t in range(N, 0, -1):
        required_mod = (mods[t-1] + mods[N]) % M
        total_case2 += freq2.get(required_mod, 0)
        current_mod = mods[t-1]
        freq2[current_mod] += 1
    
    total = total_case1 + total_case2
    print(total)

if __name__ == '__main__':
    main()