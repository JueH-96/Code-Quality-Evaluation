def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    Q = int(input_data[0])
    idx = 1
    A = []
    
    # We'll process each query
    for _ in range(Q):
        query_type = int(input_data[idx])
        idx += 1
        
        if query_type == 1:
            x = int(input_data[idx])
            idx += 1
            A.append(x)
        else:  # query_type == 2
            k = int(input_data[idx])
            idx += 1
            print(A[-k])

def main():
    solve()

if __name__ == "__main__":
    main()