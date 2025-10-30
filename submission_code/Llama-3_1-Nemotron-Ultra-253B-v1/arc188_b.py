import sys
import math

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx += 2
        if N % 2 == 0:
            M = N // 2
            if K == M:
                print("Yes" if N == 2 else "No")
            else:
                if math.gcd(K, M) == 1:
                    print("Yes")
                else:
                    print("No")
        else:
            print("Yes")

if __name__ == "__main__":
    main()