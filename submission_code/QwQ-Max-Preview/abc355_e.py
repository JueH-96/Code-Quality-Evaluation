import sys

def trailing_zeros(n):
    if n == 0:
        return 0
    cnt = 0
    while (n & 1) == 0:
        cnt += 1
        n >>= 1
    return cnt

def main():
    n, L, R = map(int, sys.stdin.readline().split())
    current = L
    b = R
    queries = []
    
    while current <= b:
        remaining_length = b - current + 1
        if remaining_length == 0:
            break
        i1 = remaining_length.bit_length() - 1
        i2 = trailing_zeros(current)
        i = min(i1, i2)
        size = 1 << i
        j = current >> i
        queries.append((i, j))
        current += size
    
    sum_mod = 0
    for i, j in queries:
        print(f"? {i} {j}")
        sys.stdout.flush()
        t = int(sys.stdin.readline())
        if t == -1:
            return
        sum_mod += t
    
    sum_mod %= 100
    print(f"! {sum_mod}")

if __name__ == "__main__":
    main()