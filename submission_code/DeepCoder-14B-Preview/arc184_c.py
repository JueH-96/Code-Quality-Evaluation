def count_ones(x):
    return bin(x).count('1')

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    max_i = (1 << 100) - A[-1] - 1 if A else (1 << 100) - 1
    max_count = 0
    for i in range(1, max_i + 1):
        current = 0
        for a in A:
            x = i + a
            if x < 1 or x >= (1 << 100):
                continue
            ones = count_ones(x)
            if ones % 2 == 1:
                current += 1
        if current > max_count:
            max_count = current
    print(max_count)

if __name__ == "__main__":
    main()