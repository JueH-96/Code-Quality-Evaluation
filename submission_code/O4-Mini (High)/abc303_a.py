def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    S = data[1]
    T = data[2]

    def normalize(c):
        # Map '1' and 'l' to the same
        if c == '1' or c == 'l':
            return '1'
        # Map '0' and 'o' to the same
        if c == '0' or c == 'o':
            return '0'
        # Otherwise itself
        return c

    for s_char, t_char in zip(S, T):
        if normalize(s_char) != normalize(t_char):
            print("No")
            return

    print("Yes")

# Call main to execute
main()