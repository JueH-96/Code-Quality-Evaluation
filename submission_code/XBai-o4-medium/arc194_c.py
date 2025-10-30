def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    C = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    initial_sum = 0
    deltas = []
    for a, b, c in zip(A, B, C):
        if a != b:
            delta = (b - a) * c
            deltas.append(delta)
        initial_sum += a * c
    
    deltas.sort()
    
    current_sum = initial_sum
    total_cost = 0
    
    for d in deltas:
        current_sum += d
        total_cost += current_sum
    
    print(total_cost)

if __name__ == "__main__":
    main()