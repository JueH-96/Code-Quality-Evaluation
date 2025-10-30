def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    functions = []
    idx = 2
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx + 1])
        functions.append((A, B))
        idx += 2
    
    # Sort functions: first by A descending, then by B descending
    sorted_funcs = sorted(functions, key=lambda x: (x[0], x[1]), reverse=True)
    
    # Select top K functions
    selected = sorted_funcs[:K]
    
    # Compose the functions in the order they are selected
    P = 1
    Q = 0
    for A, B in selected:
        Q = Q * A + B
        P = P * A
    
    # Final value is P * x + Q, with x = 1
    result = P * 1 + Q
    print(result)

if __name__ == "__main__":
    main()