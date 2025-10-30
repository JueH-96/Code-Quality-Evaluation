def main():
    import sys
    data = sys.stdin.read().splitlines()
    q = int(data[0])
    A = []
    output_lines = []
    index = 1
    for _ in range(q):
        parts = data[index].split()
        index += 1
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        else:
            k = int(parts[1])
            output_lines.append(str(A[-k]))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()