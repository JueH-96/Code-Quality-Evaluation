import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # T_mod is the total circumference modulo M
    T_mod = sum(A) % M

    freq = {}    # freq[x] = number of times prefix-sum mod M == x seen so far
    ans = 0
    P_mod = 0    # current prefix-sum mod M

    for a in A:
        x = P_mod
        # Case 1: no wrap (start < end): we count j<i with same B[j] == B[i]
        ans += freq.get(x, 0)
        # Case 2: wrap (start > end): we count j<i with B[j] == B[i] - T_mod (mod M)
        tgt = (x - T_mod) % M
        ans += freq.get(tgt, 0)

        # mark this prefix-sum for future pairs
        freq[x] = freq.get(x, 0) + 1

        # advance prefix sum
        P_mod = (P_mod + a) % M

    print(ans)

if __name__ == "__main__":
    main()