def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    prev_h = 0
    prev_u = -10**18
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        idx +=2
        if x == 0:
            option_h2 = prev_u + y if prev_u != -10**18 else -10**18
            new_h = max(prev_h, option_h2)
            new_u = prev_u
        else:
            option_u1 = prev_h + y
            option_u2 = prev_u if prev_u != -10**18 else -10**18
            new_u = max(option_u1, option_u2)
            new_h = prev_h
        prev_h, prev_u = new_h, new_u
    print(max(prev_h, prev_u))

if __name__ == "__main__":
    main()