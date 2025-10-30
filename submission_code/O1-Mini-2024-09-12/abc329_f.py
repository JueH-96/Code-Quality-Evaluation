import sys

def main():
    import sys
    from sys import stdin

    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    Q = int(data[ptr]); ptr += 1
    C = list(map(int, data[ptr:ptr+N])); ptr += N

    boxes = [{} for _ in range(N)]
    for i in range(N):
        boxes[i][C[i]] = 1

    output = []
    for _ in range(Q):
        a = int(data[ptr]) - 1
        ptr += 1
        b = int(data[ptr]) - 1
        ptr += 1

        if boxes[a]:
            if len(boxes[a]) > len(boxes[b]):
                a, b = b, a
            for color, cnt in boxes[a].items():
                if color in boxes[b]:
                    boxes[b][color] += cnt
                else:
                    boxes[b][color] = cnt
            boxes[a].clear()
        output.append(str(len(boxes[b])))

    print('
'.join(output))

if __name__ == '__main__':
    main()