def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    # Split the schedule into blocks separated by '0's
    blocks = []
    current_block = []
    for ch in S:
        if ch == '0':
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(ch)
    if current_block:
        blocks.append(current_block)
    
    max_logo_needed = 0
    for block in blocks:
        logo_needed = 0
        for day in block:
            if day == '2':
                logo_needed += 1
        max_logo_needed = max(max_logo_needed, logo_needed)
    
    print(max_logo_needed)

if __name__ == "__main__":
    main()