import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1

    scores = [0] * N
    cnt = {0: N}
    distinct = 1
    output = []

    for _ in range(T):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        player = A - 1
        old = scores[player]

        # Update old
        cnt[old] -= 1
        if cnt[old] == 0:
            del cnt[old]
            distinct -= 1

        # Update new
        new = old + B
        if new in cnt:
            cnt[new] += 1
        else:
            cnt[new] = 1
            distinct += 1

        scores[player] = new
        output.append(str(distinct))

    print('
'.join(output))

if __name__ == "__main__":
    main()