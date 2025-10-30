import sys

def main():
    # Read the sequences A, B, C
    n_a = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    m_b = int(sys.stdin.readline())
    b = list(map(int, sys.stdin.readline().split()))
    l_c = int(sys.stdin.readline())
    c = list(map(int, sys.stdin.readline().split()))
    
    # Precompute all possible sums
    sum_set = set()
    for ai in a:
        for bi in b:
            for ci in c:
                sum_set.add(ai + bi + ci)
    
    # Read the queries
    q = int(sys.stdin.readline())
    x_list = list(map(int, sys.stdin.readline().split()))
    
    # Process each query
    for x in x_list:
        print("Yes" if x in sum_set else "No")

if __name__ == "__main__":
    main()