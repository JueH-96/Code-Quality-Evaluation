def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    N = int(input_data[0])
    D = int(input_data[1])
    S = list(input_data[2])
    
    # Simulate D days. On each day, remove the cookie in the rightmost box that contains one.
    # That is achieved by iterating from right to left, and removing until D cookies have been removed.
    days_left = D
    for i in range(N - 1, -1, -1):
        if days_left == 0:
            break
        if S[i] == '@':
            S[i] = '.'
            days_left -= 1

    sys.stdout.write("".join(S))

if __name__ == '__main__':
    main()