import sys

def main():
    # Read input
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    m = int(sys.stdin.readline())
    b = list(map(int, sys.stdin.readline().split()))
    l = int(sys.stdin.readline())
    c = list(map(int, sys.stdin.readline().split()))
    q = int(sys.stdin.readline())
    xs = list(map(int, sys.stdin.readline().split()))
    
    # Precompute all possible sums of a + b + c
    sum_abc = set()
    for ai in a:
        for bi in b:
            for ci in c:
                sum_abc.add(ai + bi + ci)
    
    # Process each query
    for x in xs:
        if x in sum_abc:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()