import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    # Build segment tree
    size = 1
    while size < N:
        size <<= 1
    tree = [float('inf')] * (2 * size)
    for i in range(N):
        tree[size + i] = A[i]
    for i in range(size - 1, 0, -1):
        tree[i] = min(tree[2 * i], tree[2 * i + 1])

    # Process each query
    for b in B:
        current_l = 0
        current_r = N - 1
        node = 1
        while current_l < current_r:
            mid = (current_l + current_r) // 2
            left_node = 2 * node
            right_node = 2 * node + 1
            if tree[left_node] <= b:
                node = left_node
                current_r = mid
            else:
                node = right_node
                current_l = mid + 1
        if A[current_l] <= b:
            print(current_l + 1)
        else:
            print(-1)

if __name__ == '__main__':
    main()