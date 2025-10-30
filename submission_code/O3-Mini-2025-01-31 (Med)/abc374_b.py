def main():
    import sys
    input_lines = sys.stdin.read().splitlines()
    if len(input_lines) < 2:
        return
    S = input_lines[0].strip()
    T = input_lines[1].strip()
    
    # Iterate over the strings until we reach the end of one of them.
    min_length = min(len(S), len(T))
    for i in range(min_length):
        if S[i] != T[i]:
            print(i + 1)
            return
    
    # If one string is longer than the other.
    if len(S) != len(T):
        print(min_length + 1)
    else:
        print(0)

if __name__ == '__main__':
    main()