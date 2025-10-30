def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    strings = data[1:]
    
    # Iterate over all pairs of distinct indices.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            combined = strings[i] + strings[j]
            if combined == combined[::-1]:
                sys.stdout.write("Yes")
                return
    sys.stdout.write("No")

if __name__ == '__main__':
    main()