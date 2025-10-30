import sys

MOD = 998244353

def main() -> None:
    s = sys.stdin.readline().strip()
    n = len(s)

    # inverses of 1 .. 26
    inv = [0] * 27
    for k in range(1, 27):
        inv[k] = pow(k, MOD - 2, MOD)

    # states
    S0 = [0] * 27        # duplicates not yet happened, u different '?'-uppercases
    S0[0] = 1
    S1 = 0               # duplicate happened, no lowercase after it yet
    S2 = 0               # duplicate & already a lowercase, no uppercase after it

    fixed_seen = [False] * 26   # fixed uppercase letters already occurred
    f = 0                       # their count

    for ch in s:
        S0n = [0] * 27
        S1n = 0
        S2n = 0

        if ch == '?':                          # -------------------   '?'
            for u in range(27 - f):
                val = S0[u]
                if not val:
                    continue
                # lowercase choice
                S0n[u] = (S0n[u] + val * 26) % MOD
                # uppercase duplicate choice
                dup_letters = f + u
                S1n = (S1n + val * dup_letters) % MOD
                # uppercase new letter choice
                new_letters = 26 - f - u
                if new_letters:
                    S0n[u + 1] = (S0n[u + 1] + val * new_letters) % MOD

            # transitions for old S1 , S2
            S1n = (S1n + S1 * 26) % MOD              # uppercase keeps S1
            S2n = (S2n + S1 * 26) % MOD              # lowercase moves to S2
            S2n = (S2n + S2 * 26) % MOD              # lowercase keeps S2

        elif 'a' <= ch <= 'z':                       # --------------- lowercase
            for u in range(27 - f):
                S0n[u] = (S0n[u] + S0[u]) % MOD
            S2n = (S2 + S1) % MOD    # S1 -> S2, S2 stays
            # S1 becomes empty, S2n already set

        else:                                        # --------------- fixed uppercase
            idx = ord(ch) - 65  # 0 .. 25
            seen = fixed_seen[idx]

            if seen:                                 # second fixed occurrence
                add = sum(S0) % MOD
                S1n = (S1 + add) % MOD
                # S0 all vanish, S2 discarded
            else:
                R = 26 - f
                invR = inv[R]
                for u in range(27 - f):
                    val = S0[u]
                    if not val:
                        continue
                    dup = val * u % MOD * invR % MOD   # C already chosen from '?'
                    S1n = (S1n + dup) % MOD
                    S0n[u] = (S0n[u] + val - dup) % MOD
                S1n = (S1n + S1) % MOD    # S1 keeps
                # S2 invalid after uppercase, therefore 0
                fixed_seen[idx] = True
                f += 1

        S0, S1, S2 = S0n, S1n % MOD, S2n % MOD

    ans = (sum(S0) + S1 + S2) % MOD
    print(ans)

if __name__ == "__main__":
    main()