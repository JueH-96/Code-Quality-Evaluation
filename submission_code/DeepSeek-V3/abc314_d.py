# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    S = list(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    operations = []
    for _ in range(Q):
        t = int(data[idx])
        idx += 1
        x = int(data[idx])
        idx += 1
        c = data[idx]
        idx += 1
        operations.append((t, x, c))
    
    # Initialize the last type of operation (2 or 3) that affects the entire string
    last_global_op = None
    # To handle the order of operations, we need to process the operations in reverse
    # and keep track of the last global operation that affects the entire string
    # and apply it at the end.
    
    # We will process the operations in reverse order
    # and keep track of the last global operation
    # and also keep track of the individual character changes
    # that are not overridden by a global operation.
    
    # We will use a dictionary to store the character changes
    # that are not overridden by a global operation.
    char_changes = {}
    
    for i in range(Q-1, -1, -1):
        t, x, c = operations[i]
        if t == 1:
            if x not in char_changes:
                char_changes[x] = c
        else:
            if last_global_op is None:
                last_global_op = t
    
    # Apply the last global operation if any
    if last_global_op is not None:
        if last_global_op == 2:
            S = [s.lower() for s in S]
        elif last_global_op == 3:
            S = [s.upper() for s in S]
    
    # Apply the character changes
    for x, c in char_changes.items():
        S[x-1] = c
    
    # Now, process the operations in order to handle the remaining global operations
    # that were not the last one
    # We need to reinitialize the last_global_op
    last_global_op = None
    for i in range(Q):
        t, x, c = operations[i]
        if t == 1:
            if last_global_op is not None:
                if last_global_op == 2:
                    S[x-1] = c.lower()
                elif last_global_op == 3:
                    S[x-1] = c.upper()
                else:
                    S[x-1] = c
            else:
                S[x-1] = c
        else:
            last_global_op = t
            if t == 2:
                S = [s.lower() for s in S]
            elif t == 3:
                S = [s.upper() for s in S]
    
    print(''.join(S))

if __name__ == "__main__":
    main()