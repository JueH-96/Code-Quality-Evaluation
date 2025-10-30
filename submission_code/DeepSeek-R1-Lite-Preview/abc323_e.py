MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    T1 = T[0]
    max_s = X + T1
    freq = {}
    durations = []
    for t in T:
        if t in freq:
            freq[t] += 1
        else:
            freq[t] = 1
            durations.append(t)
    # Sort durations for efficiency
    durations.sort()
    # Initialize DP
    DP = [0] * (max_s + 1)
    DP[0] = 1
    inv_N = pow(N, MOD - 2, MOD)
    for s in range(1, max_s + 1):
        total = 0
        for t in durations:
            if s >= t:
                total = (total + freq[t] * DP[s - t]) % MOD
            else:
                break
        DP[s] = total * inv_N % MOD
    # Now compute DP1[s] for s from T1 to max_s
    DP1 = [0] * (max_s + 1)
    for s in range(T1, max_s + 1):
        DP1[s] = DP[s - T1] * inv_N % MOD
    # Sum DP1[s] from s = X + 1 to s = X + T1
    answer = 0
    for s in range(X + 1, X + T1 + 1):
        if s <= max_s:
            answer = (answer + DP1[s]) % MOD
    print(answer)

if __name__ == '__main__':
    main()