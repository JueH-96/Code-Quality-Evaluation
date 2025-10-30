import sys

def main():
    n = int(sys.stdin.readline())
    s = list(sys.stdin.readline().strip())
    q = int(sys.stdin.readline())

    modified = [False] * n
    modified_chars = [None] * n
    current_base = 0  # 0: no change, 2: lowercase, 3: uppercase

    for _ in range(q):
        t, x, c = sys.stdin.readline().split()
        x = int(x) - 1  # convert to 0-based index

        if t == 1:
            modified[x] = True
            modified_chars[x] = c
        else:
            if t == 2:
                current_base = 2
            else:
                current_base = 3

    result = []
    for i in range(n):
        if modified[i]:
            result.append(modified_chars[i])
        else:
            if current_base == 2:
                result.append(s[i].lower())
            elif current_base == 3:
                result.append(s[i].upper())
            else:
                result.append(s[i])
    
    print(''.join(result))

if __name__ == '__main__':
    main()