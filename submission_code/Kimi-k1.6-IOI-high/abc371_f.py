import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = int(input[ptr])
    ptr += 1
    queries = []
    for _ in range(Q):
        T = int(input[ptr]) - 1  # convert to 0-based index
        G = int(input[ptr + 1])
        queries.append((T, G))
        ptr += 2
    
    Y = X.copy()
    total = 0
    
    def adjust_left(T):
        nonlocal total
        j = T - 1
        while j >= 0 and Y[j] >= Y[j + 1]:
            new_val = Y[j + 1] - 1
            total += Y[j] - new_val
            Y[j] = new_val
            j -= 1
    
    def adjust_right(T):
        nonlocal total
        j = T + 1
        while j < N and Y[j] <= Y[j - 1]:
            new_val = Y[j - 1] + 1
            total += new_val - Y[j]
            Y[j] = new_val
            j += 1
    
    for T, G in queries:
        prev_val = Y[T]
        if prev_val == G:
            continue  # no change needed
        Y[T] = G
        total += abs(G - prev_val)
        # Adjust left
        adjust_left(T)
        # Adjust right
        adjust_right(T)
    
    print(total)

if __name__ == "__main__":
    main()