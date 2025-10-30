def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])      # length of the string
    S = data[1]           # the string itself
    Q = int(data[2])      # number of operations
    
    # Read the operations
    transformations = []
    idx = 3
    for _ in range(Q):
        c_i = data[idx]
        d_i = data[idx+1]
        transformations.append((c_i, d_i))
        idx += 2
    
    # A helper to convert 'a'..'z' to 0..25
    def char2idx(c):
        return ord(c) - ord('a')
    
    # A helper to convert 0..25 back to 'a'..'z'
    def idx2char(i):
        return chr(i + ord('a'))
    
    # Initialize a mapping from each character to itself
    # M[i] = i means: initially 'a' -> 'a', ..., 'z' -> 'z'
    M = list(range(26))
    
    # Process the transformations in reverse order
    # If we have c -> d in the i-th operation, we set M[c] = M[d]
    # so that after all transformations, c ends up as whatever d eventually leads to.
    for c_i, d_i in reversed(transformations):
        M[char2idx(c_i)] = M[char2idx(d_i)]
    
    # Construct the final string by mapping each character in S
    result_chars = []
    for ch in S:
        result_chars.append(idx2char(M[char2idx(ch)]))
    
    # Print the final result
    print("".join(result_chars))

# Do not forget to call main() here.
if __name__ == "__main__":
    main()