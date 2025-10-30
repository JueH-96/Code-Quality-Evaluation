import sys

def main():
    N, L, R = map(int, sys.stdin.readline().split())
    blocks = split_interval(L, R)
    total = 0
    for start, end in blocks:
        size = end - start + 1
        i = (size).bit_length() - 1
        j = start >> i
        print(f"? {i} {j}")
        sys.stdout.flush()
        res = int(sys.stdin.readline())
        if res == -1:
            return
        total += res
    print(f"! {total % 100}")
    sys.stdout.flush()

def split_interval(a, b):
    blocks = []
    while a <= b:
        best_i = -1
        best_x = -1
        best_size = 0
        max_size = b - a + 1
        for i in range(max_size.bit_length(), -1, -1):
            size = 1 << i
            if size > max_size:
                continue
            max_x = b - size + 1
            if max_x < a:
                continue
            x = (a + size - 1) // size * size
            if x <= max_x:
                best_i = i
                best_size = size
                best_x = x
                break
        if best_i == -1:
            return []
        blocks.append((best_x, best_x + best_size - 1))
        left_a, left_b = a, best_x - 1
        right_a, right_b = best_x + best_size, b
        if left_a <= left_b:
            a, b = left_a, left_b
            continue
        if right_a <= right_b:
            a, b = right_a, right_b
            continue
        break
    return blocks

if __name__ == "__main__":
    main()