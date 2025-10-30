import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    for _ in range(t):
        n = int(input[ptr])
        k = int(input[ptr + 1])
        ptr += 2
        s = input[ptr]
        ptr += 1
        B_positions = []
        for i, c in enumerate(s):
            if c == 'B':
                B_positions.append(i)
        if not B_positions:
            print(0)
            continue
        count = 0
        last_end = -1
        for b in B_positions:
            if b > last_end:
                end = b + (k - 1)
                if end >= n:
                    end = n - 1
                count += 1
                last_end = end
        print(count)

if __name__ == "__main__":
    main()