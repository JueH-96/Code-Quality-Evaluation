def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    garbage_rules = []
    for _ in range(N):
        q_i, r_i = map(int, input().split())
        garbage_rules.append((q_i, r_i))

    Q = int(input().strip())
    for _ in range(Q):
        t, d = map(int, input().split())
        q_i, r_i = garbage_rules[t - 1]

        remainder = d % q_i
        if remainder == r_i:
            print(d)
        elif remainder < r_i:
            print(d + (r_i - remainder))
        else:
            print(d + (q_i - remainder + r_i))

# Do not forget to call main()
if __name__ == "__main__":
    main()