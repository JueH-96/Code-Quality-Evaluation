def main():
    import sys
    
    S = sys.stdin.readline().strip()
    # Build a mapping from character to its position (1-based index)
    pos = {char: i+1 for i, char in enumerate(S)}
    
    total_distance = 0
    # Start at 'A'
    prev = pos['A']
    # Move through 'B' to 'Z'
    for code in range(ord('B'), ord('Z')+1):
        curr_char = chr(code)
        curr = pos[curr_char]
        total_distance += abs(curr - prev)
        prev = curr
    
    print(total_distance)

if __name__ == "__main__":
    main()