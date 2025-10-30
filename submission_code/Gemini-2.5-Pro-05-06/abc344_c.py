import sys

def solve():
    # Read N and sequence A
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read M and sequence B
    M = int(sys.stdin.readline())
    B = list(map(int, sys.stdin.readline().split()))
    
    # Read L and sequence C
    L = int(sys.stdin.readline())
    C = list(map(int, sys.stdin.readline().split()))
    
    # Read Q (number of queries) and sequence X (target sums)
    Q_count = int(sys.stdin.readline())
    X_targets = list(map(int, sys.stdin.readline().split()))

    # Precompute all possible sums a+b and store them in a set.
    # This uses a set comprehension, which is Pythonic and efficient.
    # The number of pairs (val_a, val_b) is N*M.
    # The size of sums_ab, denoted |sums_ab|, is at most N*M.
    sums_ab = {val_a + val_b for val_a in A for val_b in B}
    
    # Precompute all possible sums (a+b)+c using sums_ab and C.
    # The number of pairs (val_sab, val_c) is |sums_ab|*L.
    # The size of possible_sums_abc is at most |sums_ab|*L, which is at most N*M*L.
    possible_sums_abc = {val_sab + val_c for val_sab in sums_ab for val_c in C}
    
    # For each target sum X_i, check if it's in our set of precomputed sums.
    # Results are collected in a list and printed together at the end.
    output_lines = []
    for x_val in X_targets:
        if x_val in possible_sums_abc:
            output_lines.append("Yes")
        else:
            output_lines.append("No")
            
    # Print all results, each on a new line.
    # Adding a final newline character to match typical output expectations.
    sys.stdout.write("
".join(output_lines) + "
")

# This ensures solve() is called only when the script is executed directly.
if __name__ == '__main__':
    solve()