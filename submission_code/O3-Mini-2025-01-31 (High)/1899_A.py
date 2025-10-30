def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    t = int(input_data[0])
    results = []
    for i in range(t):
        n = int(input_data[i + 1])
        # Vanya can win immediately when n % 3 == 1 or 2 (by adding or subtracting 1 to get a multiple of 3)
        # But if n % 3 == 0, then Vanya cannot win on his move.
        if n % 3 == 0:
            results.append("Second")
        else:
            results.append("First")
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()