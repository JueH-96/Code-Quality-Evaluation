def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())
    
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            _, i, x = query
            A[i - 1] = x  # Adjust for 0-indexing
        elif query[0] == 2:
            _, i, x = query
            B[i - 1] = x  # Adjust for 0-indexing
        else:  # Type 3 query
            _, l, r = query
            result = process_type3_query(A, B, l, r)
            print(result)

def process_type3_query(A, B, l, r):
    v = 0
    for i in range(l, r + 1):  # i goes from l to r
        add_option = v + A[i - 1]  # Convert to 0-indexed
        mul_option = v * B[i - 1]  # Convert to 0-indexed
        v = max(add_option, mul_option)
    return v

if __name__ == "__main__":
    main()