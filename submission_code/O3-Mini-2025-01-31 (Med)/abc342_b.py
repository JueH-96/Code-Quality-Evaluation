def main():
    import sys
    input_data = sys.stdin.read().split()
    iterator = iter(input_data)
    
    n = int(next(iterator))
    people = [int(next(iterator)) for _ in range(n)]
    
    # Create a mapping from person number to their position in the line (0-indexed)
    position = {person: idx for idx, person in enumerate(people)}
    
    q = int(next(iterator))
    results = []
    
    for _ in range(q):
        a = int(next(iterator))
        b = int(next(iterator))
        # The one with a lower position index is further to the front
        if position[a] < position[b]:
            results.append(str(a))
        else:
            results.append(str(b))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()