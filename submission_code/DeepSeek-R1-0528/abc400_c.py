import math

def main():
    n = int(input().strip())
    k1 = math.isqrt(n // 2)
    k2 = math.isqrt(n // 4)
    print(k1 + k2)

if __name__ == '__main__':
    main()