import sys

def main():
    import sys
    N = int(sys.stdin.readline())
    buildings = []
    for _ in range(N):
        x, h = map(int, sys.stdin.readline().split())
        buildings.append((x, h))
    
    # Binary search between 0 and the maximum possible h (e.g., 1e18)
    left = 0.0
    right = 1e18
    eps = 1e-12
    required_iterations = 100  # Sufficient for precision
    
    for _ in range(required_iterations):
        mid = (left + right) / 2
        if is_possible(mid, buildings):
            right = mid
        else:
            left = mid
    
    minimal_h = (left + right) / 2
    if minimal_h <= 0:
        print(-1)
    else:
        print("{0:.20f}".format(minimal_h))

def is_possible(h, buildings):
    max_val = -float('inf')
    for x, h_i in buildings:
        required_y = h + x * max_val
        if required_y > h_i + 1e-12:  # Avoid floating point errors
            return False
        current_val = (h_i - h) / x
        if current_val > max_val:
            max_val = current_val
    return True

if __name__ == "__main__":
    main()