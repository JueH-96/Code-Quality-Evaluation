def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    # Parse inputs
    N = int(data[0])
    M = int(data[1])
    fireworks = list(map(int, data[2:2+M]))

    # We'll use a pointer to track the next fireworks day.
    j = 0
    results = []
    for i in range(1, N+1):
        # Move the pointer until fireworks[j] is not less than day i.
        while j < M and fireworks[j] < i:
            j += 1
        # The fireworks on day fireworks[j] are the first that occur on or after day i.
        results.append(str(fireworks[j] - i))
    
    sys.stdout.write("
".join(results))

# Don't forget to call main() at the end.
if __name__ == '__main__':
    main()