import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    for _ in range(T):
        N = int(input[ptr])
        ptr += 1
        A = list(map(int, input[ptr:ptr+2*N]))
        ptr += 2*N
        positions = [[] for _ in range(N+1)]
        for idx in range(2*N):
            num = A[idx]
            positions[num].append(idx)
        valid_pairs = set()
        for i in range(2*N - 1):
            current = A[i]
            next_ = A[i+1]
            if current == next_:
                continue
            a = min(current, next_)
            b = max(current, next_)
            pos_a = positions[a]
            if abs(pos_a[0] - pos_a[1]) == 1:
                continue
            pos_b = positions[b]
            if abs(pos_b[0] - pos_b[1]) == 1:
                continue
            if current == a:
                a_in = i
            else:
                a_in = i + 1
            if positions[a][0] == a_in:
                a_other = positions[a][1]
            else:
                a_other = positions[a][0]
            if current == a:
                b_in = i + 1
            else:
                b_in = i
            if positions[b][0] == b_in:
                b_other = positions[b][1]
            else:
                b_other = positions[b][0]
            if abs(a_other - b_other) == 1:
                valid_pairs.add((a, b))
        results.append(len(valid_pairs))
    for res in results:
        print(res)

if __name__ == "__main__":
    main()