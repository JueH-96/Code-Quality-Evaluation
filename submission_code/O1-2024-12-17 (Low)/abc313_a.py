def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = list(map(int, data[1:]))

    p1 = P[0]
    other_max = max(P[1:]) if N > 1 else 0  # If there's only one person, other_max is 0 by default
    
    # The required extra points for person 1 to be strictly greater than every other person's score:
    # x >= 0 such that p1 + x > other_max
    # => x > other_max - p1
    # => x = max(0, other_max + 1 - p1)
    extra_points = max(0, other_max + 1 - p1)
    
    print(extra_points)

if __name__ == "__main__":
    main()