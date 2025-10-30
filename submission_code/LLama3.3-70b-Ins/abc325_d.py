import sys

def read_int():
    return int(sys.stdin.readline().strip())

def read_ints():
    return map(int, sys.stdin.readline().strip().split())

def main():
    n = read_int()
    products = []
    for _ in range(n):
        t, d = read_ints()
        products.append((t, t + d))

    products.sort(key=lambda x: x[1])

    count = 0
    last_print_time = -1

    for t, d in products:
        if t >= last_print_time + 1:
            count += 1
            last_print_time = t

    print(count)

if __name__ == "__main__":
    main()