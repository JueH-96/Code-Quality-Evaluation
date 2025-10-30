import itertools
from math import factorial

def main():
    # Read input
    n, t, m = map(int, input().split())
    incompatibles = [tuple(map(int, input().split())) for _ in range(m)]
    
    # Convert player numbers to 0-based index
    incompatibles = [(a-1, b-1) for a, b in incompatibles]
    
    total = 0
    
    # Generate all possible color assignments (team assignments)
    for colors in itertools.product(range(1, t+1), repeat=n):
        # Check if all incompatible pairs are in different colors
        valid = True
        for a, b in incompatibles:
            if colors[a] == colors[b]:
                valid = False
                break
        if not valid:
            continue
        
        # Check if all t colors are used
        if len(set(colors)) != t:
            continue
        
        total += 1
    
    # Calculate the answer by dividing by t! to account for indistinguishable team labels
    answer = total // factorial(t)
    print(answer)

if __name__ == "__main__":
    main()