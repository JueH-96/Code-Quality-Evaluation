import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    
    # Compute prefix sums mod M and total sum mod M
    B = [0] * N  # B[i] = S_{i+1} mod M where S_1 = 0
    s = 0
    for i in range(N):
        B[i] = s
        s = (s + A[i]) % M
    R = s  # total sum mod M

    # 1) Count pairs s < t with (S_t - S_s) mod M == 0  => B[s] == B[t]
    from collections import Counter
    cnt = Counter(B)
    ans1 = 0
    for v in cnt.values():
        if v >= 2:
            ans1 += v * (v - 1) // 2

    # 2) Count pairs s > t (i.e. t=i, s=j with i<j) with
    #    (S_t - S_s + T) mod M == 0
    # => (B[i] - B[j] + R) mod M == 0
    # => B[j] == (B[i] + R) mod M
    # We rephrase: for each j in 0..N-1, count i<j with B[i] == (B[j] - R) mod M
    freq = [0] * M
    ans2 = 0
    for j in range(N):
        need = (B[j] - R) % M
        ans2 += freq[need]
        freq[B[j]] += 1

    print(ans1 + ans2)

if __name__ == "__main__":
    main()