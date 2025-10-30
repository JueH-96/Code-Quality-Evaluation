def main():
    import sys
    input_lines = sys.stdin.read().strip().split()
    if not input_lines:
        return
    N = int(input_lines[0])
    S = input_lines[1]
    
    seen = set()
    count = 0
    for char in S:
        count += 1
        seen.add(char)
        if len(seen) == 3:
            print(count)
            return

if __name__ == '__main__':
    main()