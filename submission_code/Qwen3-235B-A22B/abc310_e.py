def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = input[1]
    
    prev_0 = 0
    prev_1 = 0
    total = 0
    
    for c in S:
        current = int(c)
        new_0 = prev_1 * (current == 1)
        new_1 = prev_0 + prev_1 * (current == 0)
        if current == 0:
            new_0 += 1
        else:
            new_1 += 1
        total += new_1
        prev_0, prev_1 = new_0, new_1
    
    print(total)

if __name__ == "__main__":
    main()