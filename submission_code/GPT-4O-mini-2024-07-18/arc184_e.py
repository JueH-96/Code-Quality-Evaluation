def compute_f(A, M):
    # Function to compute the transformation of a sequence
    def transform(seq):
        new_seq = [0] * M
        current_sum = 0
        for k in range(M):
            current_sum += seq[k]
            new_seq[k] = current_sum % 2
        return new_seq

    # Precompute the transformations until they stabilize
    transformed = [A[i][:] for i in range(len(A))]
    seen = {}
    for i in range(len(A)):
        seq = tuple(transformed[i])
        steps = 0
        while seq not in seen:
            seen[seq] = steps
            transformed[i] = transform(transformed[i])
            seq = tuple(transformed[i])
            steps += 1

    # Now we can compute f(i, j)
    f_values = {}
    for i in range(len(A)):
        for j in range(i, len(A)):
            if (i, j) not in f_values:
                if transformed[i] == transformed[j]:
                    f_values[(i, j)] = 0
                else:
                    # Find the number of steps to make them equal
                    steps_i = seen[tuple(A[i])]
                    steps_j = seen[tuple(A[j])]
                    f_values[(i, j)] = abs(steps_i - steps_j)

    return f_values

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read the sequences
    A = [list(map(int, line.split())) for line in data[1:N + 1]]
    
    # Compute f(i, j) values
    f_values = compute_f(A, M)
    
    # Calculate the final sum
    total_sum = 0
    for (i, j), value in f_values.items():
        total_sum += value
    
    # Output the result modulo 998244353
    print(total_sum % 998244353)

if __name__ == "__main__":
    main()