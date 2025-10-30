def main():
    import sys
    data = sys.stdin.read().splitlines()
    q = int(data[0])
    A = []
    output_lines = []
    for i in range(1, q + 1):
        parts = data[i].split()
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        elif parts[0] == '2':
            k = int(parts[1])
            output_lines.append(str(A[-k]))
    print("
".join(output_lines))

if __name__ == "__main__":
    main()