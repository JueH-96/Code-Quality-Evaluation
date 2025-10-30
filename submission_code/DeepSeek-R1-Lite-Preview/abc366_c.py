import sys

def main():
    input = sys.stdin.read().split()
    Q = int(input[0])
    ptr = 1
    bag = {}
    output = []
    for _ in range(Q):
        op = input[ptr]
        if op == '1':
            x = int(input[ptr+1])
            bag[x] = bag.get(x, 0) + 1
            ptr += 2
        elif op == '2':
            x = int(input[ptr+1])
            bag[x] -= 1
            if bag[x] == 0:
                del bag[x]
            ptr += 2
        elif op == '3':
            output.append(str(len(bag)))
            ptr += 1
    print('
'.join(output))

if __name__ == '__main__':
    main()