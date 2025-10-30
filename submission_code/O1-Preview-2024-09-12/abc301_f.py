# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    S = sys.stdin.readline().strip()
    MOD = 998244353

    N = len(S)

    q = S.count('?')
    total_sequences = pow(52, q, MOD)

    # Preprocess the number of ways each position can be uppercase or lowercase
    upp = [0] * N
    low = [0] * N
    for i, ch in enumerate(S):
        if ch == '?':
            upp[i] = 26
            low[i] = 26
        elif 'A' <= ch <= 'Z':
            upp[i] = 1
            low[i] = 0
        elif 'a' <= ch <= 'z':
            upp[i] = 0
            low[i] = 1

    DP0 = 1  # Number of ways to process positions up to i without starting the pattern
    DP1 = [0] * 26  # DP1[c]: number of ways to have matched first uppercase letter c
    DP2 = [0] * 26  # DP2[c]: matched two uppercase letters c and c
    DP3 = [0] * 26  # DP3[c]: matched c, c, and a lowercase letter

    for i in range(N):
        sum_upp_low = (upp[i] + low[i]) % MOD

        DP0 = DP0 * sum_upp_low % MOD

        # Temporarily store previous values
        prev_DP1 = DP1[:]
        prev_DP2 = DP2[:]

        # Update DP1
        for c in range(26):
            ascii_c = chr(ord('A') + c)
            can_be_c = 0
            if S[i] == '?':
                can_be_c = 1
            elif S[i] == ascii_c:
                can_be_c = 1
            else:
                can_be_c = 0
            DP1[c] = (prev_DP1[c] * upp[i] + DP0 * can_be_c) % MOD

        # Update DP2
        for c in range(26):
            ascii_c = chr(ord('A') + c)
            can_be_c = 0
            if S[i] == '?':
                can_be_c = 1
            elif S[i] == ascii_c:
                can_be_c = 1
            else:
                can_be_c = 0
            DP2[c] = (prev_DP2[c] * upp[i] + prev_DP1[c] * can_be_c) % MOD

        # Update DP3
        for c in range(26):
            is_lowercase = 0
            if S[i] == '?':
                is_lowercase = 26
            elif 'a' <= S[i] <= 'z':
                is_lowercase = 1
            else:
                is_lowercase = 0
            DP3[c] = (DP3[c] + DP2[c] * is_lowercase) % MOD

    total_ddos = sum(DP3) % MOD
    answer = (total_sequences - total_ddos) % MOD
    print(answer)

threading.Thread(target=main).start()