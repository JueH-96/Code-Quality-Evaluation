def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    # Calculate the number of attacks needed.
    # We want the smallest integer n such that B*n >= A.
    # This is equivalent to n = ceil(A / B).
    # A common way to implement ceiling division for positive integers is: (A + B - 1) // B.
    
    result = (A + B - 1) // B
    
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()