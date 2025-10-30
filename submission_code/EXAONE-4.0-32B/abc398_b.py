def main():
    cards = list(map(int, input().split()))
    freq = [0] * 14  # Index 0 unused; indices 1-13 for card values
    
    for card in cards:
        freq[card] += 1
            
    distinct_triples = 0
    distinct_doubles = 0
    for i in range(1, 14):
        if freq[i] >= 3:
            distinct_triples += 1
        if freq[i] >= 2:
            distinct_doubles += 1
            
    if distinct_triples >= 1 and distinct_doubles >= 2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()