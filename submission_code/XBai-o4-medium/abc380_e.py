import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    left_parent = list(range(N + 1))  # 0..N
    right_end = list(range(N + 1))
    color = list(range(N + 1))
    count_color = [0] * (N + 1)

    for i in range(1, N + 1):
        count_color[i] = 1

    output = []

    def find_left(x):
        path = []
        while left_parent[x] != x:
            path.append(x)
            x = left_parent[x]
        for node in path:
            left_parent[node] = x
        return x

    for _ in range(Q):
        type_q = int(input[ptr])
        ptr += 1
        if type_q == 1:
            x = int(input[ptr])
            ptr += 1
            c = int(input[ptr])
            ptr += 1
            L = find_left(x)
            R = right_end[L]
            current_color = color[L]
            if current_color == c:
                continue
            size = R - L + 1
            count_color[current_color] -= size
            count_color[c] += size
            color[L] = c

            # Check left neighbor
            if L > 1:
                prev_L = find_left(L - 1)
                if color[prev_L] == c:
                    left_parent[L] = prev_L
                    right_end[prev_L] = R
                    L = prev_L

            # Check right neighbor
            if R < N:
                next_L = find_left(R + 1)
                if color[next_L] == c:
                    left_parent[next_L] = L
                    right_end[L] = right_end[next_L]
        else:
            c = int(input[ptr])
            ptr += 1
            output.append(str(count_color[c]))

    print('
'.join(output))

if __name__ == "__main__":
    main()