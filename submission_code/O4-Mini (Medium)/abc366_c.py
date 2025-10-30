def main():
    import sys
    input = sys.stdin.readline

    Q = int(input())
    MAX_X = 10**6
    counts = [0] * (MAX_X + 1)
    distinct = 0
    output = []

    for _ in range(Q):
        parts = input().split()
        typ = parts[0]
        if typ == '1':
            x = int(parts[1])
            if counts[x] == 0:
                distinct += 1
            counts[x] += 1
        elif typ == '2':
            x = int(parts[1])
            counts[x] -= 1
            if counts[x] == 0:
                distinct -= 1
        else:  # typ == '3'
            output.append(str(distinct))

    sys.stdout.write('
'.join(output))

# call main to execute the program
main()