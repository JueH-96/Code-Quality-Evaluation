import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    T = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1

    forbidden_pairs = []
    for _ in range(M):
        a = int(input[ptr]) - 1
        b = int(input[ptr + 1]) - 1
        ptr += 2
        if a > b:
            a, b = b, a
        forbidden_pairs.append((a, b))

    count = 0

    def backtrack(pos, current, current_max):
        nonlocal count
        if pos == N:
            if current_max + 1 != T:
                return
            for a, b in forbidden_pairs:
                if current[a] == current[b]:
                    return
            count += 1
            return

        max_so_far = current_max
        for s in range(max_so_far + 2):
            backtrack(pos + 1, current + [s], max(max_so_far, s))

    if N == 0:
        print(0)
    else:
        backtrack(1, [0], 0)
    print(count)

if __name__ == "__main__":
    main()