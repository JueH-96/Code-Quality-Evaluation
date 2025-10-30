def solve_case(R, B):
    total = R + B
    
    # Special case: exactly 2 pieces
    if total == 2:
        if R == 2:
            return "Yes
R 1 1
R 1 2"
        elif B == 2:
            return "Yes
B 1 1
B 2 2"
        else:  # R == 1, B == 1
            return "Yes
R 1 1
B 2 2"
    
    # For more than 2 pieces, we need even number of red pieces
    if R % 2 == 1:
        return "No"
    
    # Construct solution
    result = ["Yes"]
    
    # Place pieces in a simple pattern that works
    for i in range(R):
        row = 1 + i // 1000
        col = 1 + i % 1000
        result.append(f"R {row} {col}")
    
    for i in range(B):
        row = 1 + (R + i) // 1000
        col = 1 + (R + i) % 1000
        result.append(f"B {row} {col}")
    
    return "
".join(result)

T = int(input())
for _ in range(T):
    R, B = map(int, input().split())
    print(solve_case(R, B))