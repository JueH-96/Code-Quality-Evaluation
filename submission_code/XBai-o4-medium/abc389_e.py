import math

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    p = list(map(int, input[2:2 + n]))
    p.sort()
    total = 0
    rem = m
    for x in p:
        if rem == 0:
            break
        q = rem // x
        k = math.isqrt(q)
        total += k
        rem -= k * k * x
    print(total)

if __name__ == "__main__":
    main()