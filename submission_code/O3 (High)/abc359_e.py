import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n, *h_list = data

    # stack will keep pairs (height, length_of_run)
    stack = []
    total = 0          # current sum of the suffix–maximum sequence
    answers = []

    for h in h_list:
        length = 1     # new position j = i itself

        # replace the right-most runs whose value is smaller than h
        while stack and stack[-1][0] < h:
            v, cnt = stack.pop()
            total -= v * cnt
            length += cnt

        if stack and stack[-1][0] == h:
            # extend the existing run with the same height
            v, cnt = stack[-1]
            stack[-1] = (v, cnt + length)
        else:
            stack.append((h, length))

        total += h * length          # add contribution of the updated part
        answers.append(total + 1)    # first time A_i becomes positive

    sys.stdout.write(' '.join(map(str, answers)))

if __name__ == "__main__":
    main()