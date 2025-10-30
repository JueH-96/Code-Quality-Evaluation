import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    pointer = 1
    for _ in range(t):
        n = int(data[pointer])
        k = int(data[pointer + 1])
        s = data[pointer + 2]
        pointer += 3
        black_positions = [i for i in range(n) if s[i] == 'B']
        if not black_positions:
            print(0)
        else:
            operations = 0
            i = 0
            while i < len(black_positions):
                pos = black_positions[i]
                end = pos + k - 1
                if end >= n:
                    end = n - 1
                while i < len(black_positions) and black_positions[i] <= end:
                    i += 1
                operations += 1
            print(operations)

if __name__ == "__main__":
    main()