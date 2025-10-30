def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    a = list(map(int, input_data[1:1+n]))
    operations = []

    # For each index i, if a[i] is not i+1, swap it with the correct position.
    # This approach fixes each cycle with (cycle_length - 1) swaps.
    for i in range(n):
        while a[i] != i + 1:
            correct_index = a[i] - 1
            operations.append((i + 1, correct_index + 1))  # convert to 1-indexed positions
            a[i], a[correct_index] = a[correct_index], a[i]

    # Output the number of operations and the operations themselves.
    output_lines = [str(len(operations))]
    for op in operations:
        output_lines.append(f"{op[0]} {op[1]}")
    
    sys.stdout.write("
".join(output_lines))


if __name__ == "__main__":
    main()