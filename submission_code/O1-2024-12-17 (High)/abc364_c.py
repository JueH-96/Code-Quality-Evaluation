def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, X, Y = map(int, data[:3])
    A = list(map(int, data[3:3+N]))
    B = list(map(int, data[3+N:3+2*N]))
    
    dishes = list(zip(A, B))
    
    # A helper function to find how quickly (in number of dishes) 
    # we exceed X or Y when dishes are eaten in the order arr.
    def earliest_stop(arr, X, Y):
        sA = 0
        sB = 0
        for i, (a, b) in enumerate(arr):
            sA += a
            sB += b
            if sA > X or sB > Y:
                return i + 1  # 1-based index
        return len(arr)  # If we never exceed X or Y, we end up eating all
    
    # 1) Sort by A descending, then see how soon we exceed X or Y
    arrA = sorted(dishes, key=lambda x: x[0], reverse=True)
    kA = earliest_stop(arrA, X, Y)
    
    # 2) Sort by B descending, then see how soon we exceed X or Y
    arrB = sorted(dishes, key=lambda x: x[1], reverse=True)
    kB = earliest_stop(arrB, X, Y)
    
    # The answer is the minimum of the two scenarios
    print(min(kA, kB))

# Call main() to comply with the requirement
if __name__ == "__main__":
    main()