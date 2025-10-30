import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    Q = int(next(it))

    stack = [0] * 100          # initial 100 cards labeled 0
    outputs = []

    for _ in range(Q):
        t = next(it)           # query type (1 or 2)

        if t == '1':           # push x
            x = int(next(it))
            stack.append(x)
        else:                  # pop and record value
            outputs.append(str(stack.pop()))

    sys.stdout.write('
'.join(outputs))

if __name__ == "__main__":
    main()