# YOUR CODE HERE
def main():
    import sys
    lines = sys.stdin.read().splitlines()
    if len(lines) < 2:
        # Handle if all inputs are in a single line
        parts = lines[0].split()
        N, c1, c2 = parts[0], parts[1], parts[2]
        S = parts[3]
    else:
        N, c1, c2 = lines[0].split()
        S = lines[1].strip()
    
    # Replace every character that is not c1 with c2
    result = ''.join([c1 if ch == c1 else c2 for ch in S])
    print(result)

if __name__ == "__main__":
    main()