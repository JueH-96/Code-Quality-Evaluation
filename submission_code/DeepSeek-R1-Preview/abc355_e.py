import sys

def main():
    n, L, R = map(int, sys.stdin.readline().split())
    current = L
    sum_mod = 0
    while current <= R:
        max_i = 0
        for i in range(n, -1, -1):
            size = 1 << i
            if current % size == 0:
                if size <= (R - current + 1):
                    max_i = i
                    break
        j = current // (1 << max_i)
        print(f'? {max_i} {j}')
        sys.stdout.flush()
        T = int(sys.stdin.readline())
        if T == -1:
            return
        sum_mod = (sum_mod + T) % 100
        current += (1 << max_i)
    print(f'! {sum_mod}')
    sys.stdout.flush()

if __name__ == "__main__":
    main()