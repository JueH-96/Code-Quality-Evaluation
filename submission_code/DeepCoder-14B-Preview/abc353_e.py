def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]

    # Initialize the root of the trie
    root = {'children': {}, 'cnt': 0}

    for s in strings:
        current = root
        for c in s:
            if c not in current['children']:
                current['children'][c] = {'children': {}, 'cnt': 0}
            current = current['children'][c]
            current['cnt'] += 1

    total = 0
    stack = [root]

    while stack:
        current = stack.pop()
        for c, child in current['children'].items():
            stack.append(child)
            total += (child['cnt'] * (child['cnt'] - 1)) // 2

    print(total)

if __name__ == '__main__':
    main()