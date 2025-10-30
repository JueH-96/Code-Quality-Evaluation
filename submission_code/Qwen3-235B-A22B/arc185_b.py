import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr + N]))
        ptr += N
        # Compute prefix sums S
        S = [0] * (N + 1)
        for i in range(N):
            S[i + 1] = S[i] + A[i]
        valid = True
        T_prev = 0
        for k in range(1, N + 1):
            if k == 1:
                T_current = S[1]
            else:
                denominator = k - 1
                numerator = T_prev * k
                # Calculate ceil division
                min_by_avg = (numerator + denominator - 1) // denominator
                T_current = max(S[k], min_by_avg)
            # Check if T_current exceeds total sum
            if T_current > S[N]:
                valid = False
                break
            T_prev = T_current
        if valid and T_prev == S[N]:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()