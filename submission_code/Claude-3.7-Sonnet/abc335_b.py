def main():
    # Read the input N
    N = int(input())
    
    # Generate all valid triples in lexicographical order
    for x in range(N + 1):
        for y in range(N - x + 1):
            for z in range(N - x - y + 1):
                print(x, y, z)

if __name__ == "__main__":
    main()