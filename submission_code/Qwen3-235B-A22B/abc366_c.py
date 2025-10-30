import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    Q = int(input[ptr])
    ptr += 1
    counts = {}
    for _ in range(Q):
        t = input[ptr]
        if t == '1' or t == '2':
            x = int(input[ptr + 1])
            if t == '1':
                counts[x] = counts.get(x, 0) + 1
            else:
                counts[x] -= 1
                if counts[x] == 0:
                    del counts[x]
            ptr += 2
        else:
            print(len(counts))
            ptr += 1

if __name__ == "__main__":
    main()