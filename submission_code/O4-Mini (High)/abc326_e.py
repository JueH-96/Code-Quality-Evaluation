import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    mod = 998244353
    invN = pow(N, mod-2, mod)

    # P[x] = probability (modulo mod) that we ever hit state x.
    # We keep S = sum_{k=0..x-1} P[k], starting with P[0]=1 so S=1 for x=1.
    S = 1
    ans = 0
    for a in A:
        # Compute P[x] = S / N  (modularly)
        P = S * invN % mod
        # Add contribution A[x] * P[x]
        ans = (ans + P * a) % mod
        # Update prefix sum
        S = (S + P) % mod

    print(ans)

if __name__ == "__main__":
    main()