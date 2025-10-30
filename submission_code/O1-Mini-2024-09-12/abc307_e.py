import sys

def main():
    mod = 998244353
    input_data = sys.stdin.read().split()
    N, M = int(input_data[0]), int(input_data[1])
    
    term = 1 if N % 2 == 0 else mod - 1
    result = (pow(M - 1, N, mod) + term * (M - 1)) % mod
    print(result)

if __name__ == "__main__":
    main()