import sys
from math import factorial

def generate_partitions(n, k, max_part=None):
    if k == 1:
        if n >= 0:
            yield [n]
        return
    if max_part is None:
        max_part = n
    if max_part < 1:
        return
    for i in range(min(max_part, n), 0, -1):
        for p in generate_partitions(n - i, k - 1, min(i, n - i)):
            if len(p) == k - 1 and all(x <= i for x in p):
                yield [i] + p

def is_non_increasing(partition):
    for i in range(len(partition) - 1):
        if partition[i] < partition[i + 1]:
            return False
    return True

def compute_hook_product(partition):
    hook = 1
    rows = len(partition)
    for i in range(rows):
        row_length = partition[i]
        for j in range(row_length):
            right = row_length - j - 1
            below = 0
            for k in range(i + 1, rows):
                if partition[k] > j:
                    below += 1
            hook_length = right + below + 1
            hook *= hook_length
    return hook

def main():
    A, B, M = map(int, sys.stdin.readline().split())
    N = A * B - 1
    total = 0
    max_n = N
    fact = {i: factorial(i) for i in range(max_n + 1)}
    for parts in generate_partitions(N, B):
        if len(parts) != B:
            continue
        if parts[0] != A:
            continue
        if not is_non_increasing(parts):
            continue
        insertions = 0
        for i in range(1, B):
            insertions += parts[i-1] - parts[i]
        if insertions <= 0:
            continue
        hook_product = compute_hook_product(parts)
        if hook_product == 0:
            f = 0
        else:
            f = fact[N] // hook_product
        contribution = f * insertions * 2
        total += contribution
    print(total % M)

if __name__ == '__main__':
    main()