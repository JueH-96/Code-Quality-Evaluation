def main():
    import sys
    input = sys.stdin.readline
    
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Put A's elements into a set for O(1) membership checks
    a_set = set(A)
    
    # Merge and sort all elements
    C = sorted(A + B)
    
    # Check for two consecutive elements in C that both come from A
    for i in range(len(C) - 1):
        if C[i] in a_set and C[i + 1] in a_set:
            print("Yes")
            return
    
    # If no such pair found
    print("No")

if __name__ == "__main__":
    main()