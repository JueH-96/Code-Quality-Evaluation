def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    pos = []
    for i, c in enumerate(S):
        if c == '1':
            pos.append(i)
    m = len(pos)
    transformed = [pos[i] - i for i in range(m)]
    median = transformed[m // 2]
    sum_abs = sum(abs(t - median) for t in transformed)
    print(sum_abs)

if __name__ == '__main__':
    main()