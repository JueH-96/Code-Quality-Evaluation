import sys
import threading

def main():
    import sys
    A,B,M = map(int, sys.stdin.readline().split())
    # We consider the unique partition lambda of size N = A*B - 1
    # with lambda_1 = A, length(lambda) = B, sum = AB-1.
    # That partition is [A, A, ..., A, A-1] with B parts.
    N = A*B - 1
    # build lambda
    lam = [A] * (B-1) + [A-1]
    # compute lambda' (the column‐lengths) for j=1..A
    # lamp[j] = number of i with lam[i] >= j
    lamp = [0]*(A+1)
    for j in range(1, A+1):
        cnt = 0
        for x in lam:
            if x >= j:
                cnt += 1
        lamp[j] = cnt

    # hook‐product H = prod_{i=1..B, j=1..lam[i]} (lam[i]-j + lamp[j]-i +1)
    H = 1
    for i in range(1, B+1):
        li = lam[i-1]
        for j in range(1, li+1):
            hook = (lam[i-1] - j) + (lamp := lamp[j]) - i + 1
            H = (H * hook) % M

    # factorial of N mod M
    fact = 1
    for i in range(1, N+1):
        fact = fact * i % M

    # f^lambda = N! / H  (mod M)
    # then answer = 2 * f^lambda  (mod M)
    # since there are exactly two removable corners in that shape
    invH = pow(H, M-2, M)
    f = fact * invH % M
    ans = (2 * f) % M
    print(ans)

if __name__ == "__main__":
    main()