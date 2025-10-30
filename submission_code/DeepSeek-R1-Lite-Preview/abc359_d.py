from collections import defaultdict

def main():
    MOD = 998244353
    N, K = map(int, input().split())
    S = input().strip()
    
    dp_prev = defaultdict(int)
    dp_prev[''] = 1  # Initial state: empty string

    for i in range(N):
        dp_next = defaultdict(int)
        possible_cs = [S[i]] if S[i] in {'A', 'B'} else ['A', 'B']
        for s, count in dp_prev.items():
            for c in possible_cs:
                s_new = (s + c)[-K+1:]  # Keep last K-1 characters
                if i >= K-1:
                    substring = s + c
                    if substring != substring[::-1]:
                        dp_next[s_new] = (dp_next[s_new] + count) % MOD
                else:
                    dp_next[s_new] = (dp_next[s_new] + count) % MOD
        dp_prev = dp_next
    
    result = sum(dp_prev.values()) % MOD
    print(result)

if __name__ == "__main__":
    main()