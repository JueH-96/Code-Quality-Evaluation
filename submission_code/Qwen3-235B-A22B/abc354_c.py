def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    cards = []
    index = 1
    for i in range(n):
        a = int(data[index])
        c = int(data[index + 1])
        cards.append((a, c, i + 1))
        index += 2
    
    # Sort cards in descending order of A
    cards.sort(reverse=True, key=lambda x: x[0])
    
    min_c = float('inf')
    kept_indices = []
    
    for a, c, idx in cards:
        if c < min_c:
            kept_indices.append(idx)
            min_c = c
    
    kept_indices.sort()
    print(len(kept_indices))
    print(' '.join(map(str, kept_indices)))

if __name__ == '__main__':
    main()