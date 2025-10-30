def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # Precompute all j in [1..9] that divide N, along with the step size d = N/j
    divs = [(j, N // j) for j in range(1, 10) if N % j == 0]
    # Build the output string s_0..s_N
    out = []
    for i in range(N + 1):
        ch = '-'
        for j, d in divs:
            if i % d == 0:
                ch = str(j)
                break
        out.append(ch)
    sys.stdout.write(''.join(out))

# Call main to execute
main()