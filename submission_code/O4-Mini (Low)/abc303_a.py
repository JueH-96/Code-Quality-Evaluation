def main():
    import sys

    input = sys.stdin.readline
    
    N = int(input().strip())
    S = input().strip()
    T = input().strip()

    for s_char, t_char in zip(S, T):
        if s_char == t_char:
            continue
        # check the ambiguous pairs
        if (s_char == '1' and t_char == 'l') or (s_char == 'l' and t_char == '1'):
            continue
        if (s_char == '0' and t_char == 'o') or (s_char == 'o' and t_char == '0'):
            continue
        # none of the similarity conditions met
        print("No")
        return

    print("Yes")

if __name__ == "__main__":
    main()