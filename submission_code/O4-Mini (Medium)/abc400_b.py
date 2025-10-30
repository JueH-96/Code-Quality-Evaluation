import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    LIMIT = 10**9
    
    # Special case: N == 1
    # Sum = 1 + 1 + ... + 1 (M+1 terms) = M+1
    if N == 1:
        total = M + 1
        if total <= LIMIT:
            print(total)
        else:
            print("inf")
        return
    
    # General case: accumulate powers until we exceed LIMIT
    total = 1  # N^0
    term = 1   # current power of N
    for _ in range(M):
        term *= N
        # If the next term itself exceeds limit, or
        # adding it pushes total over the limit, we can stop early.
        if term > LIMIT or total + term > LIMIT:
            print("inf")
            return
        total += term
    
    # If we finish the loop without exceeding LIMIT, output the total
    print(total)

if __name__ == "__main__":
    main()