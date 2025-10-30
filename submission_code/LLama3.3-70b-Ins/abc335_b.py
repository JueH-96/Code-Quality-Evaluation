import sys

def print_triples(N):
    """
    Prints all triples of non-negative integers (x,y,z) such that x+y+z <= N 
    in ascending lexicographical order.
    """
    for x in range(N + 1):
        for y in range(N + 1):
            for z in range(N + 1):
                if x + y + z <= N:
                    print(x, y, z)

def main():
    # Read input from stdin
    N = int(sys.stdin.readline().strip())
    
    # Print triples
    print_triples(N)

if __name__ == "__main__":
    main()