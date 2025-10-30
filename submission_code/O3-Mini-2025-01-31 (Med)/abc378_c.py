def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    last_occurrence = {}
    result = []
    for i in range(n):
        value = A[i]
        # Check if value was seen before; if yes, get its last occurrence (1-indexed)
        if value in last_occurrence:
            result.append(last_occurrence[value])
        else:
            result.append(-1)
        # Update the last occurrence for the current value to current position (1-indexed)
        last_occurrence[value] = i + 1

    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()