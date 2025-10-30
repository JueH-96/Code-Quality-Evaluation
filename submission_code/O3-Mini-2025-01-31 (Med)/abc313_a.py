def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    p = list(map(int, input_data[1:]))
    
    # If there's only one person, no additional points are needed.
    if n == 1:
        print(0)
        return
    
    p1 = p[0]
    # The strongest other person's score among persons 2..N
    max_other = max(p[1:])
    
    # Calculate the additional points needed so that p1 becomes greater than max_other
    additional_points = 0
    if p1 <= max_other:
        additional_points = max_other - p1 + 1
    
    print(additional_points)

if __name__ == '__main__':
    main()