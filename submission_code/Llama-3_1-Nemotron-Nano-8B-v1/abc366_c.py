import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    Q = int(input[ptr])
    ptr += 1
    counts = defaultdict(int)
    present = set()
    output = []
    for _ in range(Q):
        query = input[ptr]
        if query == '3':
            output.append(str(len(present)))
            ptr += 1
        else:
            x = int(input[ptr + 1])
            ptr += 2
            if query == '1':
                prev = counts[x]
                counts[x] += 1
                if prev == 0:
                    present.add(x)
            else:
                counts[x] -= 1
                if counts[x] == 0:
                    present.remove(x)
    print('
'.join(output))

if __name__ == "__main__":
    main()