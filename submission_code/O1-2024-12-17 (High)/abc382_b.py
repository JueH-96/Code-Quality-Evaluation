def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])   # Number of boxes
    D = int(data[1])   # Number of days
    S = list(data[2])  # Initial state of boxes as a list of characters

    # For each of the D days, remove one cookie from the rightmost box that contains a cookie
    for _ in range(D):
        # Find the rightmost '@'
        for i in range(N - 1, -1, -1):
            if S[i] == '@':
                # Remove (eat) this cookie
                S[i] = '.'
                break

    # Print the final state after D days
    print("".join(S))

# Do not forget to call main
main()