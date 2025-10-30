import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        if N % 2 != 0:
            print("No")
        else:
            M = N // 2
            K_mod = K % M
            if math.gcd(K_mod, M) == 1:
                print("Yes")
            else:
                print("No")

if __name__ == "__main__":
    main()