# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    s = data[1]
    
    MOD = 998244353
    
    # Count the number of '1's in the string s
    k = s.count('1')
    
    # Total number of edges is N + k
    total_edges = N + k
    
    # Calculate the number of distinct sequences modulo 998244353
    result = pow(2, total_edges, MOD)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()