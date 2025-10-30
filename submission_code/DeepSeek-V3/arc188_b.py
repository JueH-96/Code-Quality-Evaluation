import sys
import math

def can_color_all(N, K):
    # Calculate the greatest common divisor (GCD) of N and K
    gcd = math.gcd(N, K)
    # If the GCD is 1, then all points can be colored black
    return gcd == 1

def main():
    input = sys.stdin.read
    data = input().split()
    T = int(data[0])
    index = 1
    for _ in range(T):
        N = int(data[index])
        K = int(data[index+1])
        index += 2
        if can_color_all(N, K):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()