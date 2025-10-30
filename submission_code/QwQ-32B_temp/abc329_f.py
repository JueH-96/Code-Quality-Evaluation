import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    C = list(map(int, input[idx:idx+N]))
    idx += N

    boxes = [set() for _ in range(N + 1)]  # 1-based indexing
    for i in range(1, N + 1):
        boxes[i].add(C[i-1])

    for _ in range(Q):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        a_set = boxes[a]
        b_set = boxes[b]
        if not a_set:
            print(len(b_set))
            continue
        # Move all elements from a to b
        b_set.update(a_set)
        a_set.clear()
        print(len(b_set))

if __name__ == "__main__":
    main()