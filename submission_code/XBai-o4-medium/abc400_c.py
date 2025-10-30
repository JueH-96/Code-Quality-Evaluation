import math

def count_good_integers(n):
    s1 = math.isqrt(2 * n) // 2
    s2 = math.isqrt(n) // 2
    return s1 + s2

def main():
    n = int(input().strip())
    print(count_good_integers(n))

if __name__ == "__main__":
    main()