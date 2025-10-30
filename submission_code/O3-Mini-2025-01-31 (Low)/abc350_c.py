def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # To quickly find positions, we use a dictionary mapping value -> index.
    pos = {A[i]: i for i in range(n)}
    
    operations = []
    
    for i in range(n):
        correct_value = i + 1
        if A[i] != correct_value:
            # index of the correct value
            j = pos[correct_value]
            
            # swap A[i] and A[j]
            # record 1-indexed swap indices in the answer
            operations.append((i+1, j+1))
            
            # update the dictionary for swapped elements
            pos[A[i]] = j
            pos[A[j]] = i
            
            # perform swap in the list
            A[i], A[j] = A[j], A[i]
    
    # Output result
    output_lines = []
    output_lines.append(str(len(operations)))
    for op in operations:
        output_lines.append(f"{op[0]} {op[1]}")
    
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()